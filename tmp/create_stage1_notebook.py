"""Create the Stage 1 launcher without modifying the frozen source/data bundle."""

import copy
import json
from pathlib import Path

source = Path('notebooks/05_kaggle_kgat_resume.ipynb')
original = source.read_bytes()
notebook = copy.deepcopy(json.loads(original))


def set_cell(index, text):
    notebook['cells'][index]['source'] = text.strip().splitlines(keepends=True)


set_cell(0, '''
# T-KGAT Stage 1 — thời gian trên cạnh user–item

Dùng nguyên Dataset code/data và ZIP Recovery **KGAT đã hoàn thành 20/20 lượt,
có kết quả test đủ 5 seed**. Không cần chờ KGAT không KG.
Notebook chỉ chạy `tkgat.yaml`, tự đọc cấu hình KGAT đã chọn trên validation.
Quét đủ **γ = {1, 3, 10} × 5 seed 2019–2023 = 15 lượt**.
Chọn γ theo Recall@20 trung bình trên validation; chỉ cấu hình được chọn mới chấm test.

Mặc định chạy tối đa 2 lượt mới mỗi phiên, lưu ZIP sau mỗi lượt hoàn tất.
Chọn **Save Version → Save & Run All** với GPU T4. Chờ log
`Running tkgat Stage 1 only` rồi có thể đóng tab/tắt máy.
''')
set_cell(1, '''
from pathlib import Path

PAYLOAD_DIR = None
# Select a ZIP or results directory only if automatic discovery finds multiple recoveries.
RESTORE_SOURCE = None
PROJECT_ROOT = Path('/kaggle/working/t-kgat-stage1')
RUN_TRAINING = True
NEW_TKGAT_FITS_THIS_SESSION = 2
CHECKPOINT_ZIP = Path('/kaggle/working/tkgat-lastfm-stage1-results.zip')
''')
set_cell(2, '''
## 1. Kiểm tra source/data và khôi phục KGAT

Phiên đầu dùng `tkgat-lastfm-results.zip` hoàn chỉnh của KGAT có KG.
Phiên sau dùng ZIP `tkgat-lastfm-stage1-results.zip` mới nhất của chính nhánh Stage 1.
Giữ một bản Recovery trong Input; nếu có nhiều bản, chỉ định `RESTORE_SOURCE`.
ZIP hoặc thư mục `results/` đã giải nén đều được hỗ trợ.
''')

preparation = ''.join(notebook['cells'][3]['source'])
guard = '''
def require_stage1_inputs(suite):
    """Validate the completed KGAT dependency and the full Stage 1 gamma/seed grid."""
    require_baselines(suite)
    if not suite['complete']('kgat'):
        raise RuntimeError('KGAT is incomplete in Recovery. Use the completed KGAT ZIP before Stage 1.')
    print('kgat: complete (verified)', flush=True)
    import yaml

    model_path = suite['ROOT'] / 'configs/model/tkgat.yaml'
    config = yaml.safe_load(model_path.read_text(encoding='utf-8'))
    if (config.get('name') != 'tkgat' or config.get('time_on', 'interact') != 'interact'
            or config.get('control') or config.get('from_kgat') != 'kgat'):
        raise RuntimeError('Expected the unpermuted Stage 1 configuration from KGAT.')
    if config.get('seeds') != list(range(2019, 2024)):
        raise RuntimeError('Stage 1 requires all five seeds 2019-2023.')
    if config.get('grid') != [{'gamma': 1}, {'gamma': 3}, {'gamma': 10}]:
        raise RuntimeError('Stage 1 requires the complete gamma grid [1, 3, 10].')
    runner = runpy.run_path(str(suite['ROOT'] / 'scripts/run.py'))
    folder = suite['ROOT'] / 'results/runs' / suite['RUN_NAME']
    points, base = runner['tkgat_points'](config, model_path, folder)
    print('Inherited KGAT configuration:', json.dumps(base, sort_keys=True), flush=True)
    print('Stage 1 grid: gamma=[1, 3, 10], seeds=[2019, 2020, 2021, 2022, 2023]; 15 fits.', flush=True)
    return points


'''
preparation = preparation.replace("input_root = Path('/kaggle/input')", guard + "input_root = Path('/kaggle/input')")
preparation = preparation.replace("require_baselines(suite)\nsuite['backup']", "require_stage1_inputs(suite)\nsuite['backup']")
preparation = preparation.replace('Recovery verified. KGAT is the only model enabled.',
                                  'Recovery verified. T-KGAT Stage 1 is the only model enabled.')
preparation = preparation.replace('Refusing to restart MostPop/BPR-MF.', 'Refusing to start Stage 1 without KGAT Recovery.')
set_cell(3, preparation)

set_cell(4, '''
## 2. Tiếp tục lưới T-KGAT Stage 1

Log phải xác nhận `kgat: complete (verified)` và hiển thị lưới đầy đủ 15 lượt.
Thứ tự mặc định: γ=1 đủ 5 seed, rồi γ=3, rồi γ=10.
Lượt đã hoàn thành được đọc từ checkpoint; lượt bị ngắt giữa epoch phải train lại.
Mỗi lần gọi hoàn thành tối đa 1 lượt mới rồi lưu ZIP.
Sau đủ 15 lượt, script chọn γ trên validation và chấm test 5 seed của γ đã chọn.
Không tự chạy đối chứng hoán vị hay Stage 2 trong notebook này.
''')

training = ''.join(notebook['cells'][5]['source'])
training = training.replace('require_baselines(suite)', 'require_stage1_inputs(suite)')
training = training.replace('NEW_KG_FITS_THIS_SESSION', 'NEW_TKGAT_FITS_THIS_SESSION')
training = training.replace("suite['complete']('kgat')", "suite['complete']('tkgat')")
training = training.replace('configs/model/kgat.yaml', 'configs/model/tkgat.yaml')
training = training.replace('Running kgat only;', 'Running tkgat Stage 1 only;')
training = training.replace("print('kgat:", "print('tkgat:")
set_cell(5, training)

set_cell(6, '''
## 3. Lưu kết quả và chạy phiên sau

Sau khi Version thành công, tải **`tkgat-lastfm-stage1-results.zip`** trong Output.
Thay Recovery cũ bằng ZIP Stage 1 mới nhất trước khi chạy Version kế tiếp.
Giữ Dataset code/data ban đầu. Nếu chỉ giữ 2 lượt/phiên, cần tối đa 8 phiên
(7 phiên đầu mỗi phiên 2 lượt; phiên cuối hoàn tất lượt 15 rồi chọn γ và chấm test).

Khi hoàn tất, `tkgat_summary.json` ghi γ được chọn, Recall/NDCG test trung bình ±
độ lệch chuẩn mẫu trên 5 seed; ZIP giữ kết quả KGAT/baseline và toàn bộ tiến độ Stage 1.
Bước sau là đối chứng hoán vị Stage 1 và/hoặc Stage 2, dùng notebook tương ứng.
''')

for index, cell in enumerate(notebook['cells'], 1):
    cell['id'] = f'tkgat-stage1-{index}'
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
        cell['metadata'] = {}

notebook.setdefault('metadata', {})['kaggle'] = {'isGpuEnabled': True, 'isInternetEnabled': False}
target = Path('notebooks/07_kaggle_tkgat_stage1_resume.ipynb')
if target.exists():
    raise FileExistsError(target)
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
assert source.read_bytes() == original
print(target.resolve())
