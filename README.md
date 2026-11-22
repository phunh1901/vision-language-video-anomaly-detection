# CLIP-based Video Anomaly Detection

Kho mã nghiên cứu **Weakly-Supervised Video Anomaly Detection (WSVAD)** dựa trên
CLIP, lấy **VadCLIP** làm baseline và đối chứng trên **UCF-Crime**, **XD-Violence**.
Hướng nghiên cứu gồm xử lý thời gian và căn chỉnh video–văn bản, cùng khảo sát
PoinCLIP-VAD và Alert-CLIP. Khảo sát một mô hình không đồng nghĩa đã tái lập mô hình đó.

## Bài toán

Huấn luyện từ nhãn cấp video, sinh điểm bất thường theo snippet và ánh xạ về frame
để đánh giá. CLIP cung cấp biểu diễn thị giác–ngôn ngữ; VadCLIP bổ sung xử lý thời gian
và hai nhánh phân loại/căn chỉnh. Điểm phát hiện hỗ trợ rà soát video, không tự kết luận pháp lý.

Luồng thực nghiệm hiện tại dùng feature CLIP đã trích xuất. Phép đo latency từ feature
trên GPU tới logits là inference offline, chưa bao gồm giải mã video hoặc trích xuất CLIP.

## Cấu trúc repository

```text
versions/
    README.md               Danh mục và hướng dẫn chạy các version
    V00.ipynb … V21.ipynb    Notebook độc lập, mỗi version một file chuẩn
    _dev/                   Source, recipe, builder và helper dùng phát triển
results/
    README.md               Chỉ mục kết quả
    Vxx/<run>/notes.md      Kết luận, phân tích và giới hạn của run
    Vxx/<run>/metrics.json  Metric đọc từ artifact và nguồn đối chiếu
tests/                      Test số học, dữ liệu, recipe, recovery và notebook
pyproject.toml              Python/dependency và cấu hình kiểm tra
uv.lock                     Dependency lock
```

[versions/](versions/README.md) là nơi chọn code chạy.
[results/](results/README.md) là nơi xem output và kinh nghiệm.
Git chỉ nhận code, test, output văn bản đã rà soát và các file cấu hình cần để chạy;
dữ liệu, checkpoint, ZIP gốc, báo cáo cá nhân và chỉ dẫn nội bộ giữ ngoài repository.

## Các mốc thí nghiệm

| Version | Vai trò |
|---|---|
| [V00](versions/V00.ipynb) | Inference checkpoint VadCLIP V0 làm mốc đối chứng |
| [V07](versions/V07.ipynb) | Suy luận bằng cửa sổ chồng lấn |
| [V20](versions/V20.ipynb) | Centered temporal head |
| [V21](versions/V21.ipynb) | Notebook hiện hành dùng native cosine control; các recipe Poincaré giữ riêng |

V0–V15 có runtime nhúng trong notebook; việc không có thư mục Python riêng không có
nghĩa thiếu version. Source phát triển V16–V21 nằm trong `_dev`.
V21 không mặc định là tái lập đầy đủ PoinCLIP-VAD hoặc một cải tiến đã được xác nhận.

## Chạy trên Kaggle

1. Chọn `versions/Vxx.ipynb` và import lên Kaggle.
2. Đọc cell cấu hình, gắn feature/checkpoint đúng recipe và kiểm `RUN_MODE` trước chạy.
3. Dùng smoke để kiểm đường chạy khi cần; full mới đánh giá đủ split. Một số notebook đang lưu full.
4. Dùng cell cuối tải output, giữ run ID và source/config/hash để đối chiếu.

| Input | Yêu cầu |
|---|---|
| UCF-Crime test | 290 video, crop `__5.npy` theo CSV/evaluator đã khóa |
| XD-Violence test | “VadCLIP XD-Violence Test 800”, crop `__0.npy` theo evaluator |
| Checkpoint | V0 riêng cho UCF/XD; kiểm SHA-256 trong notebook |
| Train | Phụ thuộc version; V21 native cần đủ 10 crop/video: UCF 16.100 file, XD 39.540 file |

Hai gói `train_v5` một crop không đáp ứng V21 native.
Notebook standalone nhúng source nên không cần upload riêng các file `.py`.
Xem [hướng dẫn V21](versions/_dev/v21/NOTEBOOK_GUIDE_COSINE.md) trước khi chạy recipe này.

## Kiểm tra local

Python 3.12 và `uv`; môi trường local dùng PyTorch CPU:

```powershell
uv sync --frozen
uv run --frozen ruff check versions/_dev tests
uv run --frozen python -m pytest
```

Test CPU/fixture kiểm hành vi và tính nhất quán, không thay cho benchmark GPU thật.
Builder V21: `uv run --frozen python versions/_dev/v21/build_notebook.py COSINE`.
Lệnh này sinh lại `versions/V21.ipynb`; cần kiểm diff trước khi dùng bản sinh lại.

## Đọc kết quả

| Dataset | Chỉ số chính | Bổ sung |
|---|---|---|
| UCF-Crime | C-Branch frame ROC-AUC | Ano-AUC, AP và A-Branch |
| XD-Violence | A-Branch frame AP | ROC-AUC và C-Branch |

Mỗi giả thuyết đối chiếu với V0 trong điều kiện phù hợp; giữ nhánh chính cố định.
`metrics.json` giữ tên metric và giá trị gốc của JSON trong archive, thường ở thang 0–1,
kèm nguồn member và SHA-256 archive. Đọc JSON chưa phải replay độc lập từ prediction.
Run thiếu metric được ghi rõ, không điền 0 hoặc suy là chưa chạy.

ZIP gốc có thể chứa checkpoint, raw prediction và log lớn nên không nằm trong bản clone Git.
Output công khai gồm bảng metric và ghi chú kỹ thuật; chúng không thay cho toàn bộ artifact tái lập.
Official test đã được xem nhiều lần trong quá trình phát triển, vì vậy các kết luận hồi cứu
được ghi là thăm dò; giữ cả kết quả âm và các giới hạn kiểm chứng.
