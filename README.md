# CLIP-based Video Anomaly Detection

Nghiên cứu phát hiện bất thường trong video với nhãn yếu ở cấp video (WSVAD).
Pipeline dùng feature CLIP ViT-B/16 đã trích xuất và VadCLIP làm baseline, sau đó
ánh xạ điểm bất thường từ snippet về frame để đánh giá.

## Notebook Kaggle

| Bản | Phương pháp | UCF-Crime · C frame ROC-AUC | XD-Violence · A frame AP |
|---|---|---:|---:|
| [V0](versions/V00.ipynb) | Checkpoint VadCLIP công bố | 88,01993% | 84,50449% |
| [V7](versions/V07.ipynb) | Suy luận với cửa sổ chồng lấn | 88,02336% | 84,61980% |
| [V20](versions/V20.ipynb) | Head thời gian với hiệu chỉnh logit cân bằng theo video | 88,04726% | 84,81664% |

Các điểm trong bảng lấy từ cùng lượt full V20 trên 290 video UCF và 800 video XD.
So với V0, AP XD tăng 0,11531 điểm phần trăm ở V7 và 0,31216 điểm ở V20.
Biến động AUC UCF nhỏ và khoảng bootstrap 95% chứa 0. Official test đã được xem
nhiều lần trong quá trình phát triển, nên các mức tăng này là kết quả thăm dò.
Xem [bảng kết quả và nguồn](results/README.md).

## Cách chạy

1. Import notebook lên Kaggle, chọn GPU và bật Internet.
2. Gắn feature test UCF/XD và checkpoint tương ứng. V20 cần thêm hai gói feature
   train V5 để huấn luyện head. Xem [hướng dẫn đầu vào](versions/README.md).
3. Chạy `RUN_MODE = "smoke"` để kiểm tra đầu vào, rồi dùng `"full"` để chấm đủ
   hai tập. Cell cuối tải gói kết quả.

Latency đo từ feature và mask trên GPU đến logits, ngoài phạm vi giải mã video
và trích xuất feature CLIP.

## Kiểm tra local

Dùng Python 3.12 và `uv`.

```powershell
uv sync --frozen
uv run --frozen ruff check versions/_dev tests
uv run --frozen python -m pytest
```
