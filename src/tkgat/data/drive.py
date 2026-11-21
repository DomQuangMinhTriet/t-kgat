"""Read members of a Google Drive ZIP in memory, without extracting source files.

Sources have the form ``gdrive://<file-id>/<member>``. Public files use gdown;
private files can use a short-lived OAuth token in TKGAT_DRIVE_ACCESS_TOKEN.
The ChatGPT Drive connection is not inherited by standalone Python processes.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import sys
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_STREAM_RETRIES = 5


class DriveResumeError(RuntimeError):
    """The server cannot safely continue the same compressed byte stream."""


def _confirmation_url(html: str) -> str:
    from gdown.download import get_url_from_gdrive_confirmation

    url = get_url_from_gdrive_confirmation(html)
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if parsed.scheme != "https" or not (host.endswith(".google.com") or host.endswith(".googleusercontent.com")):
        raise RuntimeError("Drive confirmation returned an unexpected download host")
    return url


def _drive_response(session, file_id: str, offset: int, validator):
    """Resolve public confirmations afresh, then request the remaining bytes."""
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        if validator:
            headers["If-Range"] = validator[1]
    token = os.environ.get("TKGAT_DRIVE_ACCESS_TOKEN")
    if token:
        return session.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&supportsAllDrives=true",
            headers={**headers, "Authorization": f"Bearer {token}"}, stream=True, timeout=(15, 60),
        )

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    for _ in range(5):
        # Do not apply Range to the small HTML confirmation page.
        response = session.get(url, headers={"Accept-Encoding": "identity"}, stream=True, timeout=(15, 60))
        try:
            response.raise_for_status()
            if "text/html" not in response.headers.get("Content-Type", "").lower():
                if not offset:
                    return response
                url = response.url
                response.close()
                return session.get(url, headers=headers, stream=True, timeout=(15, 60))
            url = _confirmation_url(response.text)
        except BaseException:
            response.close()
            raise
        response.close()
    raise RuntimeError("Drive did not resolve to a downloadable file; check sharing or quota")


def _range_info(response, offset: int, total, validator):
    """Reject ignored/misaligned ranges and changed files before yielding data."""
    if response.headers.get("Content-Encoding", "identity").lower() not in {"", "identity"}:
        raise DriveResumeError("Drive returned HTTP-compressed content; byte-range resume would be unsafe")
    if offset and response.status_code != 206:
        raise DriveResumeError(f"Drive ignored resume at byte {offset} (HTTP {response.status_code}); stopping safely")
    if response.status_code == 206:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if not match:
            raise DriveResumeError("Drive returned an invalid Content-Range")
        start, end, size = map(int, match.groups())
        if start != offset or not start <= end < size:
            raise DriveResumeError(f"Drive returned the wrong byte range for offset {offset}")
        segment_end = end + 1
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != segment_end - offset:
            raise DriveResumeError("Drive Content-Length does not match its byte range")
    elif response.status_code == 200 and offset == 0:
        length = response.headers.get("Content-Length")
        size = int(length) if length is not None else None
        segment_end = size
    else:
        raise DriveResumeError(f"Unexpected Drive download response: HTTP {response.status_code}")
    if total is not None and size != total:
        raise DriveResumeError("Drive file size changed during streaming; stopping safely")
    if validator:
        current = response.headers.get(validator[0])
        if current is not None and current != validator[1]:
            raise DriveResumeError("Drive file changed during streaming; stopping safely")
    else:
        etag = response.headers.get("ETag", "")
        if etag and not etag.startswith("W/"):
            validator = ("ETag", etag)
        elif response.headers.get("Last-Modified"):
            validator = ("Last-Modified", response.headers["Last-Modified"])
    return size, segment_end, validator


def drive_chunks(file_id: str):
    """Stream bytes with bounded retries/resume; retain the live ZIP decoder."""
    import requests

    file_id = drive_file_id(file_id)
    offset, retries = 0, 0
    total = validator = None
    with requests.Session() as session:
        while True:
            response = None
            try:
                response = _drive_response(session, file_id, offset, validator)
                response.raise_for_status()
                total, segment_end, validator = _range_info(response, offset, total, validator)
                for chunk in response.iter_content(chunk_size=512 * 1024):
                    if not chunk:
                        continue
                    if segment_end is not None and offset + len(chunk) > segment_end:
                        raise DriveResumeError("Drive returned more bytes than its announced range")
                    offset += len(chunk)
                    yield chunk
                if segment_end is not None and offset < segment_end:
                    raise requests.exceptions.ChunkedEncodingError(
                        f"Read ended at byte {offset}, expected {segment_end}"
                    )
                if total is None or offset == total:
                    return
                # Some servers return only part of the requested range.
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError, requests.exceptions.HTTPError) as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status is not None and status not in {429, 500, 502, 503, 504}:
                    raise RuntimeError(
                        f"Drive rejected the download (HTTP {status}); check file access or quota"
                    ) from exc
                if retries >= MAX_STREAM_RETRIES:
                    raise RuntimeError(
                        f"Drive stream interrupted at byte {offset} after {retries} retries; "
                        "the current process cannot continue"
                    ) from exc
                retries += 1
                print(f"Drive read interrupted; reconnecting at byte {offset} (retry {retries}/{MAX_STREAM_RETRIES})",
                      file=sys.stderr, flush=True)
                if response is not None:
                    response.close()
                time.sleep(min(2 ** (retries - 1), 16))
            finally:
                if response is not None:
                    response.close()


def zip_member_chunks(file_id: str, member: str):
    """Sequentially decompress a ZIP64 member; accept a basename or exact path."""
    try:
        from stream_unzip import stream_unzip
    except ImportError as exc:
        raise RuntimeError('Install ZIP streaming support: python -m pip install -e ".[drive]"') from exc
    source = drive_chunks(file_id)
    try:
        for name, _, content in stream_unzip(source):
            name = name.decode("utf-8").replace("\\", "/")
            if name == member or ("/" not in member and name.rsplit("/", 1)[-1] == member):
                yield from content  # exhaust the member to validate its CRC
                return
            for _ in content:
                pass
        raise FileNotFoundError(f"ZIP member not found: {member}")
    finally:
        if hasattr(source, "close"):
            source.close()


class _ChunkReader(io.RawIOBase):
    def __init__(self, chunks):
        super().__init__()
        self.chunks = iter(chunks)
        self.pending = memoryview(b"")

    def readable(self):
        return True

    def readinto(self, target):
        while not self.pending:
            try:
                self.pending = memoryview(next(self.chunks))
            except StopIteration:
                return 0
        count = min(len(target), len(self.pending))
        target[:count] = self.pending[:count]
        self.pending = self.pending[count:]
        return count

    def close(self):
        try:
            if not self.closed and hasattr(self.chunks, "close"):
                self.chunks.close()
        finally:
            super().close()


class _MemoryBuffer(io.BytesIO):
    def write(self, data):
        if self.tell() + len(data) > MAX_ARCHIVE_BYTES:
            raise ValueError("Drive ZIP exceeds the 512 MiB in-memory limit")
        return super().write(data)


def drive_file_id(value: str) -> str:
    """Accept a raw file ID or the Drive file URL; folder URLs are not files."""
    if re.fullmatch(r"[\w-]+", value):
        return value
    parsed = urlsplit(value)
    match = re.fullmatch(r"/file/d/([\w-]+)(?:/.*)?", parsed.path)
    if parsed.scheme == "https" and parsed.hostname == "drive.google.com" and match:
        return match[1]
    raise ValueError("Expected a Drive ZIP file ID or https://drive.google.com/file/d/<ID>/view, not a folder")


@lru_cache(maxsize=1)
def drive_archive(file_id: str) -> ZipFile:
    """Read the compressed ZIP once per process. No ZIP or raw member is saved."""
    file_id = drive_file_id(file_id)
    buffer = _MemoryBuffer()
    try:
        token = os.environ.get("TKGAT_DRIVE_ACCESS_TOKEN")
        if token:
            request = Request(
                f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&supportsAllDrives=true",
                headers={"Authorization": f"Bearer {token}"},
            )
            with urlopen(request, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    buffer.write(chunk)
        else:
            try:
                import gdown
            except ImportError as exc:
                raise RuntimeError('Install Drive support: python -m pip install -e ".[drive]"') from exc
            try:
                result = gdown.download(id=file_id, output=buffer, quiet=False, use_cookies=False)
            except Exception as exc:
                raise RuntimeError(
                    "Cannot read the Drive ZIP publicly. For a private file, set TKGAT_DRIVE_ACCESS_TOKEN "
                    "to an OAuth token with Drive read access. The ChatGPT connection is separate."
                ) from exc
            if result is None:
                raise RuntimeError("Drive did not return the ZIP; check file access and quota")
        buffer.seek(0)
        return ZipFile(buffer)
    except BadZipFile as exc:
        buffer.close()
        raise ValueError("Drive returned invalid ZIP content (possibly a sign-in or quota page)") from exc
    except BaseException:
        buffer.close()
        raise


def verify_archive(file_id: str, expected_sha256: str) -> None:
    archive = drive_archive(file_id)
    # ZipFile retains the original BytesIO; getbuffer avoids copying the ZIP.
    with archive.fp.getbuffer() as view:
        actual = hashlib.sha256(view).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Drive ZIP SHA-256 mismatch: expected {expected_sha256}, got {actual}")


def join_source(directory: str | Path, filename: str) -> str:
    if str(directory).startswith("gdrive://"):
        return f"{str(directory).rstrip('/')}/{filename}"
    return str(Path(directory) / filename)


@contextmanager
def open_binary_source(source: str | Path):
    value = str(source)
    if value.startswith("gdrive-stream://"):
        parsed = urlsplit(value)
        member = parsed.path.lstrip("/")
        if not member or parsed.query or parsed.fragment:
            raise ValueError("Streaming sources must be gdrive-stream://<ZIP-file-id>/<member-path-or-basename>")
        handle = io.BufferedReader(_ChunkReader(zip_member_chunks(parsed.netloc, member)), buffer_size=1024 * 1024)
    elif value.startswith("gdrive://"):
        parsed = urlsplit(value)
        if not parsed.path.lstrip("/") or parsed.query or parsed.fragment:
            raise ValueError("Drive sources must be gdrive://<ZIP-file-id>/<member-path>")
        archive = drive_archive(parsed.netloc)
        member = parsed.path.lstrip("/")
        try:
            handle = archive.open(member)
        except KeyError as exc:
            raise FileNotFoundError(f"ZIP member not found: {member}") from exc
    else:
        handle = Path(source).open("rb")
    with handle:
        yield handle


@contextmanager
def open_text_source(source: str | Path):
    with open_binary_source(source) as binary, io.TextIOWrapper(binary, encoding="utf-8") as text:
        yield text
