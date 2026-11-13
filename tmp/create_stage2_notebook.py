"""Create the Stage 2 launcher using the existing frozen source/data payload."""

import ast
import copy
import json
from pathlib import Path

source = Path('notebooks/07_kaggle_tkgat_stage1_resume.ipynb')
original = source.read_bytes()
notebook = copy.deepcopy(json.loads(original))


def set_cell(index, text):
    notebook['cells'][index]['source'] = text.strip().splitlines(keepends=True)


set_cell(0, '''
# T-KGAT Stage 2 — thời gian trên cạnh KG

Dùng Dataset code/data đã đóng gói và ZIP Recovery **KGAT có KG đã hoàn tất**.
Stage 2 tự đọc cấu hình KGAT đã chọn trên validation và quét đủ
**γ = {1, 3, 10} × 5 seed 2019–2023 = 15 lượt**.
Stage 1 và Stage 2 có thể chạy song song sau khi KGAT đã hoàn thành.

Thời gian được áp dụng lên cạnh KG có ngày. Ngày KG của bộ Last-FM này được suy ra
từ Freebase. Fact không có ngày nhận phạt thời gian bằng 0; độ cũ chuẩn hóa được
chặn tối đa ở 1. Trong payload Last-FM hiện tại, toàn bộ 125.640 fact có ngày
đều bị chặn ở 1. Vì vậy Stage 2 trên bộ này chủ yếu kiểm tra tác động giảm
trọng số nhóm cạnh có ngày so với nhóm không có ngày; các fact có ngày cùng
nhận mức phạt gamma trước softmax. Cạnh user–item nhận mức phạt bằng 0.

Mặc định chạy 2 lượt mới mỗi phiên và lưu ZIP sau mỗi lượt hoàn tất.
Chọn **Save Version → Save & Run All**, bật GPU T4 và kiểm tra log
`Running tkgat Stage 2 only` trước khi đóng tab/tắt máy.
''')
set_cell(1, '''
from pathlib import Path

PAYLOAD_DIR = None
# Select a ZIP or results directory if automatic discovery finds several recoveries.
RESTORE_SOURCE = None
PROJECT_ROOT = Path('/kaggle/working/t-kgat-stage2')
RUN_TRAINING = True
NEW_TKGAT_FITS_THIS_SESSION = 2
CHECKPOINT_ZIP = Path('/kaggle/working/tkgat-lastfm-stage2-results.zip')
''')
set_cell(2, '''
## 1. Nạp code/data và khôi phục KGAT

Phiên đầu dùng ZIP `tkgat-lastfm-results.zip` có kết quả KGAT đầy đủ 20 lượt và 5 seed test.
Phiên tiếp theo dùng ZIP `tkgat-lastfm-stage2-results.zip` mới nhất của nhánh Stage 2.
Dataset code/data ban đầu đã có `kg_t1_dates.tsv` và `kg_t2_dates.tsv`.
Gắn một bản Recovery, hoặc đặt `RESTORE_SOURCE` nếu có nhiều bản trong Input.
''')

preparation = ''.join(notebook['cells'][3]['source'])
start = preparation.index('def require_baselines(suite):')
end = preparation.index('def require_stage1_inputs(suite):')
preparation = preparation[:start] + preparation[end:]
preparation = preparation.replace('require_stage1_inputs', 'require_stage2_inputs')
preparation = preparation.replace('    require_baselines(suite)\n', '')
preparation = preparation.replace('Stage 1', 'Stage 2')
preparation = preparation.replace('configs/model/tkgat.yaml', 'configs/model/tkgat_kg.yaml')
preparation = preparation.replace("config.get('time_on', 'interact') != 'interact'", "config.get('time_on') != 'kg'")
set_cell(3, preparation)

set_cell(4, '''
## 2. Chạy lưới T-KGAT Stage 2

Log phải có `kgat: complete (verified)`, lưới gamma đầy đủ và
`Running tkgat Stage 2 only`. Stage 2 gọi cấu hình `tkgat_kg.yaml` với `time_on: kg`.
Mỗi lượt huấn luyện dùng train, dừng sớm theo Recall@20 validation.
Sau đủ 15 lượt, script chọn gamma có Recall@20 validation trung bình tốt nhất qua 5 seed,
rồi chấm test 5 seed của gamma được chọn.

Các lượt hoàn tất được đọc từ checkpoint. Lượt bị ngắt giữa epoch phải chạy lại từ đầu.
Giữ kết quả của từng nhánh bằng ZIP có tên riêng khi chạy trên các máy khác nhau.
''')
training = ''.join(notebook['cells'][5]['source'])
training = training.replace('require_stage1_inputs', 'require_stage2_inputs')
training = training.replace("suite['complete']('tkgat')", "suite['complete']('tkgat_kg')")
training = training.replace('configs/model/tkgat.yaml', 'configs/model/tkgat_kg.yaml')
training = training.replace('Running tkgat Stage 1 only;', 'Running tkgat Stage 2 only;')
training = training.replace("print('tkgat:", "print('tkgat_kg:")
set_cell(5, training)

set_cell(6, '''
## 3. Lưu kết quả và chạy phiên sau

Tải **`tkgat-lastfm-stage2-results.zip`** từ Output của Version đã hoàn tất.
Dùng ZIP Stage 2 mới nhất để thay Recovery cho phiên tiếp theo, giữ Dataset code/data.
Mức 2 lượt mới/phiên tương ứng tối đa 8 phiên cho lưới 15 lượt.

Khi hoàn tất, `tkgat_kg_summary.json` ghi gamma đã chọn và điểm test trung bình ±
độ lệch chuẩn mẫu qua 5 seed. ZIP cũng giữ kết quả theo từng user và checkpoint.
Đối chứng hoán vị ngày fact sử dụng gamma Stage 2 đã chọn và sẽ chạy bằng notebook riêng.
''')

for index, cell in enumerate(notebook['cells'], 1):
    cell['id'] = f'tkgat-stage2-{index}'
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']))
        cell['outputs'] = []
        cell['execution_count'] = None
        cell['metadata'] = {}

notebook.setdefault('metadata', {})['kaggle'] = {'isGpuEnabled': True, 'isInternetEnabled': False}
target = Path('notebooks/08_kaggle_tkgat_stage2_resume.ipynb')
if target.exists():
    raise FileExistsError(target)
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
assert source.read_bytes() == original
print(target.resolve())
