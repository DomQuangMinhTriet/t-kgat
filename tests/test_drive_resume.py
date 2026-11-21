"""Interrupted HTTP streams must resume compressed bytes without loss or duplication."""

import io
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from tkgat.data import drive

requests = pytest.importorskip("requests", reason="HTTP resume tests require the drive extra")


class Response:
    def __init__(self, chunks=(), status=200, headers=None, text=""):
        self.chunks = chunks
        self.status_code = status
        self.headers = headers or {}
        self.text = text
        self.url = "https://drive.usercontent.google.com/download?id=test"
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}", response=self)

    def iter_content(self, chunk_size):
        assert chunk_size == 512 * 1024
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self):
        self.closed = True


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture
def http(monkeypatch):
    monkeypatch.setenv("TKGAT_DRIVE_ACCESS_TOKEN", "fake-token")
    sleeps = []
    monkeypatch.setattr(drive.time, "sleep", sleeps.append)

    def install(responses):
        session = Session(responses)
        monkeypatch.setattr(requests, "Session", lambda: session)
        return session

    return install, sleeps


def first(chunks, size=6):
    return Response(chunks, headers={"Content-Length": str(size), "ETag": '"version-a"'})


def rest(chunks=(b"def",), start=3, end=5, size=6, **headers):
    return Response(chunks, status=206, headers={
        "Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(end - start + 1),
        "ETag": '"version-a"', **headers,
    })


def test_resume_starts_after_last_yielded_byte_and_preserves_file_validator(http):
    install, sleeps = http
    a = first([b"abc", requests.exceptions.ChunkedEncodingError("connection broken")])
    b = rest()
    session = install([a, b])
    assert b"".join(drive.drive_chunks("test-id")) == b"abcdef"
    assert session.calls[1][1]["headers"]["Range"] == "bytes=3-"
    assert session.calls[1][1]["headers"]["If-Range"] == '"version-a"'
    assert session.calls[1][1]["headers"]["Authorization"] == "Bearer fake-token"
    assert a.closed and b.closed and sleeps == [1]


def test_silent_short_response_also_reconnects(http):
    install, _ = http
    session = install([first([b"abc"]), rest()])
    assert b"".join(drive.drive_chunks("test-id")) == b"abcdef"
    assert session.calls[1][1]["headers"]["Range"] == "bytes=3-"


@pytest.mark.parametrize("response", [
    Response([b"abcdef"], headers={"Content-Length": "6"}),  # ignored Range
    rest(start=2, end=5),  # overlapping bytes
    rest(size=7),  # file size changed
    rest(**{"ETag": '"version-b"'}),  # file replaced
    Response([b"def"], status=206),  # missing Content-Range
])
def test_bad_resume_responses_stop_before_appending_any_wrong_bytes(http, response):
    install, _ = http
    install([first([b"abc", requests.exceptions.ConnectionError("disconnected")]), response])
    stream = drive.drive_chunks("test-id")
    assert next(stream) == b"abc"
    with pytest.raises(drive.DriveResumeError):
        next(stream)
    assert response.closed


def test_retries_are_bounded_and_network_error_does_not_claim_permission_failure(http, monkeypatch):
    install, sleeps = http
    monkeypatch.setattr(drive, "MAX_STREAM_RETRIES", 2)
    session = install([requests.exceptions.Timeout("timed out") for _ in range(3)])
    with pytest.raises(RuntimeError, match="byte 0 after 2 retries"):
        list(drive.drive_chunks("test-id"))
    assert len(session.calls) == 3 and sleeps == [1, 2]


def test_permission_rejection_is_not_retried(http):
    install, sleeps = http
    response = Response(status=403)
    session = install([response])
    with pytest.raises(RuntimeError, match="HTTP 403"):
        list(drive.drive_chunks("test-id"))
    assert len(session.calls) == 1 and not sleeps and response.closed


def test_transient_server_error_is_retried(http):
    install, sleeps = http
    response = Response(status=503)
    install([response, first([b"abcdef"])])
    assert b"".join(drive.drive_chunks("test-id")) == b"abcdef"
    assert sleeps == [1] and response.closed


def test_short_valid_ranges_continue_without_being_treated_as_failures(http):
    install, sleeps = http
    session = install([rest([b"abc"], start=0, end=2), rest()])
    assert b"".join(drive.drive_chunks("test-id")) == b"abcdef"
    assert not sleeps and session.calls[1][1]["headers"]["Range"] == "bytes=3-"


def test_resumed_bytes_continue_the_existing_zip_decoder(http):
    pytest.importorskip("stream_unzip")
    install, _ = http
    expected = "user\titem\tts\n" + "".join(f"1\t2\t{ts}\n" for ts in range(1000))
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("data/events.inter", expected)
    content = buffer.getvalue()
    cut = len(content) // 3
    session = install([
        first([content[:cut], requests.exceptions.ChunkedEncodingError("broken")], size=len(content)),
        rest([content[cut:]], start=cut, end=len(content) - 1, size=len(content)),
    ])
    with drive.open_text_source("gdrive-stream://test-id/events.inter") as handle:
        assert handle.read() == expected
    assert session.calls[1][1]["headers"]["Range"] == f"bytes={cut}-"


def test_public_confirmation_is_refreshed_without_sending_range_to_html(http, monkeypatch):
    install, _ = http
    monkeypatch.delenv("TKGAT_DRIVE_ACCESS_TOKEN")
    monkeypatch.setattr(drive, "_confirmation_url", lambda html: "https://drive.usercontent.google.com/download")
    html = lambda: Response(headers={"Content-Type": "text/html"}, text="confirmation")
    probe = first([])
    session = install([
        html(), first([b"abc", requests.exceptions.ChunkedEncodingError("broken")]), html(), probe, rest(),
    ])
    assert b"".join(drive.drive_chunks("test-id")) == b"abcdef"
    for _, call in session.calls[:-1]:
        assert "Range" not in call["headers"] and "Authorization" not in call["headers"]
    assert session.calls[-1][1]["headers"]["Range"] == "bytes=3-"
    assert probe.closed


def test_confirmation_form_uses_the_existing_gdown_parser():
    pytest.importorskip("gdown")
    html = (
        '<form id="download-form" action="https://drive.usercontent.google.com/download">'
        '<input type="hidden" name="id" value="test-id"></form>'
    )
    assert drive._confirmation_url(html) == "https://drive.usercontent.google.com/download?id=test-id"


def test_response_is_closed_when_the_zip_reader_finishes_early(http):
    install, _ = http
    response = first([b"abc", b"def"])
    install([response])
    stream = drive.drive_chunks("test-id")
    assert next(stream) == b"abc"
    stream.close()
    assert response.closed
