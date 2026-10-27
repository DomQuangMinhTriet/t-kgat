# Tham số huấn luyện BPR-MF (tự sinh bởi `scripts/09_significance.py`)

Đọc từ `configs/model/bprmf.yaml` và các file history trong `results/runs/`; không sửa tay.

| Tham số | ML-100K | ML-1M |
|---|---|---|
| Batch size | 2048 | 2048 |
| Số epoch tối đa | 300 | 300 |
| Đánh giá val mỗi (epoch) | 5 | 5 |
| Patience (số lần đánh giá) | 5 | 5 |
| Chỉ số chọn | recall@20 | recall@20 |
| Learning rate | 0.001 | 0.001 |
| L2 | 0.0001 | 0.0001 |
| Dim | 32 | 64 |
| Điểm lưới được chọn trên val | 3 / 3 | 1 / 3 |
| Seed | 2019, 2020, 2021, 2022, 2023 | 2019, 2020, 2021, 2022, 2023 |
| Epoch đã chạy (nhỏ nhất đến lớn nhất qua seed) | 60 đến 135 | 40 đến 70 |
| Epoch tốt nhất (nhỏ nhất đến lớn nhất qua seed) | 35 đến 110 | 15 đến 45 |
