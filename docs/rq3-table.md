# Bảng RQ3 (tự sinh bởi `scripts/10_rq3_table.py`)

> **Số của KGAT, KGAT không KG, T-KGAT và T-KGAT hoán vị là kết quả sơ bộ, chưa đóng băng (tag `v0.2-kgat`), chạy trước khi cô duyệt giao thức** (t1/t2, quyết định 0002). Chưa có tag `v0.2-kgat`: các số này chưa đóng băng. KGAT và T-KGAT không tái lập từng bit trên GPU: coi chênh lệch giữa các lần chạy của một seed như nhiễu seed.
>
> **ml-1m-u10 là 10% user của ML-1M, không phải ML-1M đầy đủ.**

Nguồn: `results/registry.csv` qua `registry.load`, chỉ dòng `phase = test` không mang nhãn `-dirty`; mỗi mô hình lấy điểm của lần chạy mới nhất trên split mới nhất. Trung bình ± độ lệch chuẩn **mẫu** (`ddof=1`) qua các seed; một seed thì không có độ lệch chuẩn. Kiểm định: Wilcoxon cặp theo từng user (`tkgat.eval.stats`) trên trung bình các seed chung, Holm trong cùng bộ dữ liệu và metric, họ cố định gồm ba phép so sánh; hai phép của giai đoạn 2 (nếu có) là một họ riêng. KG gồm fact Wikidata (bản chụp 2026) và metadata MovieLens (quyết định 0011).

## ml-100k (user test: 51)

| Mô hình | Cấu hình | Recall@20 (test) | NDCG@20 (test) | Số seed |
|---|---|---|---|---|
| MostPop | window_days = None | 0,1555 | 0,1451 | 1 |
| MostPop recent | window_days = 14 | 0,3372 | 0,2870 | 1 |
| BPR-MF | dim = 32, l2 = 0.0001, lr = 0.001 | 0,1471 ± 0,0295 | 0,1549 ± 0,0170 | 5 |
| KGAT [sơ bộ] | l2 = 0.0001, l2_kg = 1e-05, lr = 0.001, mess_dropout = 0.1 | 0,2096 ± 0,0284 | 0,1859 ± 0,0104 | 5 |
| KGAT không KG [sơ bộ] | l2 = 0.0001, l2_kg = 1e-05, lr = 0.001, mess_dropout = 0.1 | 0,2017 ± 0,0232 | 0,1764 ± 0,0185 | 5 |
| T-KGAT [sơ bộ] | gamma = 3, l2 = 0.0001, l2_kg = 1e-05, lr = 0.001, mess_dropout = 0.1 | 0,2151 ± 0,0174 | 0,1840 ± 0,0136 | 5 |
| T-KGAT (đối chứng hoán vị) [sơ bộ] | gamma = 3, l2 = 0.0001, l2_kg = 1e-05, lr = 0.001, mess_dropout = 0.1 | 0,1950 ± 0,0156 | 0,1789 ± 0,0099 | 5 |
| T-KGAT giai đoạn 2 (Δt trên KG) [sơ bộ] | gamma = 1, l2 = 0.0001, l2_kg = 1e-05, lr = 0.001, mess_dropout = 0.1, time_on = kg | 0,1933 ± 0,0216 | 0,1817 ± 0,0079 | 5 |
| T-KGAT giai đoạn 2 (đối chứng hoán vị ngày fact) [sơ bộ] | gamma = 1, l2 = 0.0001, l2_kg = 1e-05, lr = 0.001, mess_dropout = 0.1, time_on = kg | 0,1971 ± 0,0238 | 0,1826 ± 0,0122 | 5 |

**So sánh cặp** (hiệu = trước − sau):

| So sánh | Metric | Seed chung | n_pairs | n_nonzero | Hiệu trung bình | p thô | p Holm | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| T-KGAT so với KGAT | Recall@20 | 5 | 51 | 22 | 0,0055 | 0,733 | 1,000 | only 51 evaluation users (< 100); only 22 non-zero differences (< 30) |
| T-KGAT thật so với hoán vị | Recall@20 | 5 | 51 | 29 | 0,0201 | 0,315 | 0,944 | only 51 evaluation users (< 100); only 29 non-zero differences (< 30) |
| KGAT so với KGAT không KG | Recall@20 | 5 | 51 | 28 | 0,0079 | 0,909 | 1,000 | only 51 evaluation users (< 100); only 28 non-zero differences (< 30) |
| T-KGAT giai đoạn 2 so với KGAT | Recall@20 | 5 | 51 | 30 | -0,0164 | 0,162 | 0,162 | only 51 evaluation users (< 100) |
| T-KGAT giai đoạn 2 thật so với hoán vị | Recall@20 | 5 | 51 | 26 | -0,0038 | 0,078 | 0,155 | only 51 evaluation users (< 100); only 26 non-zero differences (< 30) |
| T-KGAT so với KGAT | NDCG@20 | 5 | 51 | 35 | -0,0019 | 0,512 | 0,651 | only 51 evaluation users (< 100) |
| T-KGAT thật so với hoán vị | NDCG@20 | 5 | 51 | 34 | 0,0051 | 0,326 | 0,651 | only 51 evaluation users (< 100) |
| KGAT so với KGAT không KG | NDCG@20 | 5 | 51 | 35 | 0,0095 | 0,196 | 0,587 | only 51 evaluation users (< 100) |
| T-KGAT giai đoạn 2 so với KGAT | NDCG@20 | 5 | 51 | 35 | -0,0042 | 0,376 | 0,456 | only 51 evaluation users (< 100) |
| T-KGAT giai đoạn 2 thật so với hoán vị | NDCG@20 | 5 | 51 | 33 | -0,0009 | 0,228 | 0,456 | only 51 evaluation users (< 100) |

![Learning curve val của KGAT, T-KGAT và T-KGAT hoán vị trên ml-100k](figures/rq3_curves_ml-100k.png)

## ml-1m-u10 (user test: 95)

> ml-1m-u10 là 10% user của ML-1M, không phải ML-1M đầy đủ.

| Mô hình | Cấu hình | Recall@20 (test) | NDCG@20 (test) | Số seed |
|---|---|---|---|---|
| MostPop | window_days = None | 0,0846 | 0,1623 | 1 |
| MostPop recent | window_days = 3 | 0,1007 | 0,1619 | 1 |
| BPR-MF | dim = 32, l2 = 0.0001, lr = 0.001 | 0,1000 ± 0,0069 | 0,1889 ± 0,0102 | 5 |
| KGAT [sơ bộ] | l2 = 1e-05, l2_kg = 1e-05, lr = 0.001, mess_dropout = 0.1 | 0,1086 ± 0,0054 | 0,1873 ± 0,0139 | 5 |
| KGAT không KG [sơ bộ] | l2 = 0.0001, l2_kg = 1e-05, lr = 0.001, mess_dropout = 0.1 | 0,1042 ± 0,0101 | 0,1773 ± 0,0160 | 5 |
| T-KGAT [sơ bộ] | gamma = 1, l2 = 1e-05, l2_kg = 1e-05, lr = 0.001, mess_dropout = 0.1 | 0,1113 ± 0,0046 | 0,1942 ± 0,0056 | 5 |
| T-KGAT (đối chứng hoán vị) [sơ bộ] | gamma = 1, l2 = 1e-05, l2_kg = 1e-05, lr = 0.001, mess_dropout = 0.1 | 0,1119 ± 0,0049 | 0,1951 ± 0,0079 | 5 |
| T-KGAT giai đoạn 2 (Δt trên KG) [sơ bộ] | gamma = 1, l2 = 1e-05, l2_kg = 1e-05, lr = 0.001, mess_dropout = 0.1, time_on = kg | 0,1145 ± 0,0148 | 0,1958 ± 0,0109 | 5 |
| T-KGAT giai đoạn 2 (đối chứng hoán vị ngày fact) [sơ bộ] | gamma = 1, l2 = 1e-05, l2_kg = 1e-05, lr = 0.001, mess_dropout = 0.1, time_on = kg | 0,1164 ± 0,0099 | 0,1919 ± 0,0097 | 5 |

**So sánh cặp** (hiệu = trước − sau):

| So sánh | Metric | Seed chung | n_pairs | n_nonzero | Hiệu trung bình | p thô | p Holm | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| T-KGAT so với KGAT | Recall@20 | 5 | 95 | 66 | 0,0027 | 0,225 | 0,450 | only 95 evaluation users (< 100) |
| T-KGAT thật so với hoán vị | Recall@20 | 5 | 95 | 65 | -0,0006 | 0,360 | 0,450 | only 95 evaluation users (< 100) |
| KGAT so với KGAT không KG | Recall@20 | 5 | 95 | 75 | 0,0044 | 0,048 | 0,143 | only 95 evaluation users (< 100) |
| T-KGAT giai đoạn 2 so với KGAT | Recall@20 | 5 | 95 | 68 | 0,0058 | 0,095 | 0,189 | only 95 evaluation users (< 100) |
| T-KGAT giai đoạn 2 thật so với hoán vị | Recall@20 | 5 | 95 | 61 | -0,0020 | 0,900 | 0,900 | only 95 evaluation users (< 100) |
| T-KGAT so với KGAT | NDCG@20 | 5 | 95 | 82 | 0,0069 | 0,016 | 0,049 | only 95 evaluation users (< 100) |
| T-KGAT thật so với hoán vị | NDCG@20 | 5 | 95 | 82 | -0,0008 | 0,833 | 0,833 | only 95 evaluation users (< 100) |
| KGAT so với KGAT không KG | NDCG@20 | 5 | 95 | 84 | 0,0100 | 0,016 | 0,049 | only 95 evaluation users (< 100) |
| T-KGAT giai đoạn 2 so với KGAT | NDCG@20 | 5 | 95 | 83 | 0,0085 | 0,032 | 0,065 | only 95 evaluation users (< 100) |
| T-KGAT giai đoạn 2 thật so với hoán vị | NDCG@20 | 5 | 95 | 82 | 0,0038 | 0,276 | 0,276 | only 95 evaluation users (< 100) |

![Learning curve val của KGAT, T-KGAT và T-KGAT hoán vị trên ml-1m-u10](figures/rq3_curves_ml-1m-u10.png)

## Phụ lục: KG chỉ có Wikidata (bị thay bởi quyết định 0011)

> Các dòng này chạy trước quyết định 0011, trên KG chưa có metadata MovieLens. Chúng **không phải kết quả chính thức**; giữ lại để người đọc thấy cả hai, vì KG được đổi sau khi điểm test của chúng đã được xem.

| Bộ dữ liệu | Mô hình | Cấu hình | Recall@20 (test) | NDCG@20 (test) | Số seed |
|---|---|---|---|---|---|
| ml-100k | KGAT | l2 = 0.0001, l2_kg = 1e-05, lr = 0.001, mess_dropout = 0.1 | 0,2025 ± 0,0116 | 0,1753 ± 0,0107 | 5 |
| ml-100k | T-KGAT | gamma = 1, l2 = 0.0001, l2_kg = 1e-05, lr = 0.001, mess_dropout = 0.1 | 0,2165 ± 0,0104 | 0,1843 ± 0,0076 | 5 |
| ml-100k | T-KGAT (đối chứng hoán vị) | gamma = 1, l2 = 0.0001, l2_kg = 1e-05, lr = 0.001, mess_dropout = 0.1 | 0,1989 ± 0,0176 | 0,1754 ± 0,0061 | 5 |
| ml-1m-u10 | KGAT | l2 = 0.0001, l2_kg = 1e-05, lr = 0.001, mess_dropout = 0.1 | 0,1121 ± 0,0071 | 0,1989 ± 0,0071 | 5 |
| ml-1m-u10 | T-KGAT | gamma = 1, l2 = 0.0001, l2_kg = 1e-05, lr = 0.001, mess_dropout = 0.1 | 0,1104 ± 0,0074 | 0,1964 ± 0,0041 | 5 |
| ml-1m-u10 | T-KGAT (đối chứng hoán vị) | gamma = 1, l2 = 0.0001, l2_kg = 1e-05, lr = 0.001, mess_dropout = 0.1 | 0,1107 ± 0,0078 | 0,1956 ± 0,0049 | 5 |
