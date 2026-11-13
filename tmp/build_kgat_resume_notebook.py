"""Build the dedicated resume notebook without changing the frozen model payload."""

import json
from pathlib import Path
from textwrap import dedent


def cell(kind, source, index):
    result = {"cell_type": kind, "id": f"kgat-resume-{index}", "metadata": {},
              "source": dedent(source).lstrip().splitlines(keepends=True)}
    if kind == "code":
        result.update(outputs=[], execution_count=None)
    return result


cells = []


def add(kind, source):
    cells.append(cell(kind, source, len(cells) + 1))


add("markdown", """
# Tiếp tục KGAT trên Kaggle — Save & Run All

Notebook này **chỉ chạy KGAT**, yêu cầu MostPop và BPR-MF đã hoàn tất trong kết quả
khôi phục. Nếu thiếu hoặc sai source/data, notebook dừng báo lỗi trước khi train.
Giữ Dataset code/data cũ và thêm Dataset Recovery vào Input (Private).
Recovery có thể là ZIP hoặc thư mục `results/` đã được Kaggle giải nén.

Mặc định chạy 2 lượt cấu hình–seed mới; KGAT vẫn có đầy đủ 4 cấu hình × 5 seed.
Lượt train dở chạy lại từ đầu; chỉ các lượt đã lưu checkpoint và validation mới được tiếp tục.
Chọn **Save Version → Save & Run All**, bật GPU T4, kiểm tra log `Running kgat only`
rồi có thể đóng tab/tắt máy. Phiên chạy nền khác Draft Session.
""")

add("code", """
from pathlib import Path

PAYLOAD_DIR = None
# Optional: select a ZIP, a results directory, or its parent if several recoveries exist.
RESTORE_SOURCE = None
PROJECT_ROOT = Path('/kaggle/working/t-kgat-kgat')
RUN_TRAINING = True
NEW_KG_FITS_THIS_SESSION = 2
CHECKPOINT_ZIP = Path('/kaggle/working/tkgat-lastfm-results.zip')
""")

add("markdown", """
## 1. Nạp code/data và khôi phục kết quả

Không cần upload lại source.bundle/data. Dùng gói Recovery có kết quả BPR-MF đủ 5 seed.
Nếu có nhiều bản Recovery, đặt `RESTORE_SOURCE` thành đường dẫn bản muốn dùng.
Không tự chọn bản mới nhất và không ghi đè tiến độ đang có.
""")

add("code", r'''
import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, is_zipfile


def recovery_archive(input_root, working, expected_identity, explicit=None):
    """Resolve one matching ZIP or extracted results directory, never guess between copies."""
    if explicit is not None:
        location = Path(explicit)
        if not location.exists():
            raise RuntimeError(f'Recovery path does not exist: {location}')
        locations = [location]
    else:
        locations = sorted(input_root.rglob('*.zip'))
        locations += sorted(p.parent for p in input_root.rglob('kaggle-run.json')
                            if p.parent.name == 'results')
    matches = []
    inspected = []
    for location in locations:
        if location.is_file() and is_zipfile(location):
            with ZipFile(location) as archive:
                marker = 'results/kaggle-run.json'
                if marker not in archive.namelist():
                    continue
                saved_identity = json.loads(archive.read(marker))
            candidate = ('zip', location)
        elif location.is_dir():
            folder = location if location.name == 'results' else location / 'results'
            marker = folder / 'kaggle-run.json'
            if not marker.is_file():
                continue
            saved_identity = json.loads(marker.read_text(encoding='utf-8'))
            candidate = ('directory', folder)
        else:
            continue
        inspected.append(str(location))
        if saved_identity == expected_identity:
            matches.append(candidate)
    if not matches:
        raise RuntimeError(
            'No matching Recovery found. Add the previous results as Input; '
            'both ZIP and extracted results/ are supported. '
            f'Found result locations: {inspected}. Refusing to restart MostPop/BPR-MF.'
        )
    if len(matches) != 1:
        raise RuntimeError(f'Multiple recoveries match: {matches}. Set RESTORE_SOURCE explicitly.')
    kind, source = matches[0]
    print('Recovery selected:', source, flush=True)
    if kind == 'zip':
        return source
    target = working / 'kgat-recovery-normalized.zip'
    with ZipFile(target, 'w', compression=ZIP_DEFLATED) as archive:
        for folder in (source, source.parent / 'reports'):
            if folder.is_dir():
                for path in sorted(folder.rglob('*')):
                    if path.is_file():
                        archive.write(path, path.relative_to(source.parent).as_posix())
    return target


def require_baselines(suite):
    """Require complete summaries, unique test seeds, and all per-user outputs."""
    for model in ('mostpop', 'bprmf'):
        if not suite['complete'](model):
            raise RuntimeError(
                f'{model}: incomplete in Recovery. Add the complete previous results; '
                'this notebook only runs KGAT and will not retrain baselines.'
            )
        print(f'{model}: complete (verified)', flush=True)


input_root = Path('/kaggle/input')
working = Path('/kaggle/working')
if PAYLOAD_DIR is None:
    manifests = sorted(input_root.rglob('payload-manifest.json'))
    if not manifests:
        packages = sorted(input_root.rglob('tkgat-lastfm-latest-kaggle.zip'))
        if len(packages) != 1:
            raise RuntimeError('Add the original code/data Dataset as Input, or set PAYLOAD_DIR.')
        unpacked = working / 'tkgat-payload'
        with ZipFile(packages[0]) as archive:
            for name in archive.namelist():
                path = PurePosixPath(name)
                if path.is_absolute() or '..' in path.parts or '\\' in name:
                    raise ValueError(f'Invalid payload member: {name}')
            archive.extractall(unpacked)
        manifests = [unpacked / 'payload-manifest.json']
    if len({path.read_text(encoding='utf-8') for path in manifests}) != 1:
        raise RuntimeError('Multiple code/data payloads found. Set PAYLOAD_DIR.')
    payload_root = manifests[0].parent
else:
    payload_root = Path(PAYLOAD_DIR)
manifest = json.loads((payload_root / 'payload-manifest.json').read_text(encoding='utf-8'))
bundle = payload_root / 'source.bundle'
if hashlib.sha256(bundle.read_bytes()).hexdigest() != manifest['files_sha256']['source.bundle']:
    raise ValueError('Source bundle hash mismatch')
if not PROJECT_ROOT.exists():
    subprocess.run(['git', 'clone', str(bundle), str(PROJECT_ROOT)], check=True)
commit = subprocess.check_output(['git', '-C', str(PROJECT_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
if commit != manifest['source_commit']:
    raise RuntimeError('Working copy belongs to another payload. Start a fresh session.')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
os.environ['PYTHONPATH'] = str(PROJECT_ROOT / 'src')
os.environ['PYTHONUNBUFFERED'] = '1'
suite = runpy.run_path(str(PROJECT_ROOT / 'scripts/kaggle_lastfm.py'))
suite['prepare'](payload_root)
has_progress = (PROJECT_ROOT / 'results/registry.csv').exists() or any(
    (PROJECT_ROOT / 'results/runs').rglob('*.pt')
)
if not has_progress:
    restore = recovery_archive(input_root, working, suite['identity'](), RESTORE_SOURCE)
    suite['extract_results'](restore)
else:
    print('Keeping existing working progress; Recovery will not overwrite it.', flush=True)
suite['record_identity']()
require_baselines(suite)
suite['backup'](CHECKPOINT_ZIP)
print('Source snapshot:', commit, flush=True)
print('Recovery verified. KGAT is the only model enabled.', flush=True)
''')

add("markdown", """
## 2. Kiểm tra GPU và tiếp tục KGAT

Trước khi train, log phải có `mostpop: complete (verified)` và
`bprmf: complete (verified)`. Nếu thiếu, notebook báo lỗi và không train.
Mỗi lần gọi chỉ hoàn thành tối đa 1 lượt mới rồi cập nhật ZIP; lặp theo
`NEW_KG_FITS_THIS_SESSION`. Chỉ chấm test sau khi toàn bộ lưới đã hoàn tất.
Giới hạn thời gian/quota của Kaggle vẫn áp dụng; notebook không tự vượt giới hạn.
""")

add("code", """
import torch

require_baselines(suite)
print('CUDA:', torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise RuntimeError('Enable GPU T4 before Save & Run All.')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
if NEW_KG_FITS_THIS_SESSION < 1:
    raise ValueError('NEW_KG_FITS_THIS_SESSION must be positive.')

if RUN_TRAINING:
    for step in range(NEW_KG_FITS_THIS_SESSION):
        if suite['complete']('kgat'):
            print('kgat: complete. No other model will be started.', flush=True)
            break
        print(f'Running kgat only; fit budget {step + 1}/{NEW_KG_FITS_THIS_SESSION}', flush=True)
        try:
            suite['execute']([
                'scripts/run.py', '--data', suite['DATA_CONFIG'],
                '--model', 'configs/model/kgat.yaml', '--device', 'cuda',
                '--resume', '--max-new-runs', '1',
            ])
        finally:
            suite['backup'](CHECKPOINT_ZIP)
else:
    print('Training disabled. Recovery verified only.', flush=True)

suite['backup'](CHECKPOINT_ZIP)
print('kgat:', 'complete' if suite['complete']('kgat') else 'pending (more fits needed)', flush=True)
print('Save this output:', CHECKPOINT_ZIP, flush=True)
""")

add("markdown", """
## 3. Giữ kết quả và chạy phiên sau

Sau khi Version hoàn thành, tải `/kaggle/working/tkgat-lastfm-results.zip` trong
**Output**. Phiên sau thay Recovery cũ bằng gói kết quả mới nhất; giữ nguyên
Dataset code/data. Có thể thêm Output của Version trước làm Input.
Nếu có nhiều bản kết quả, chỉ định `RESTORE_SOURCE`; không tự chọn theo thời gian file.

ZIP chứa cả MostPop/BPR-MF cũ và tiến độ KGAT. Không chỉ tải riêng file `.pt`.
""")

notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "kaggle": {"isGpuEnabled": True, "isInternetEnabled": False}}, "nbformat": 4, "nbformat_minor": 5}
target = Path('notebooks/05_kaggle_kgat_resume.ipynb')
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(target)
