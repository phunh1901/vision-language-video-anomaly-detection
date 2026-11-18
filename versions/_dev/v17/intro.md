# VadCLIP V17 — Residual C→A và cửa sổ chồng lấn V7

So bốn cấu hình trên UCF-Crime và XD-Violence với cùng checkpoint V0:

| Cấu hình | Cửa sổ | Residual C→A |
|---|---|---|
| V0 | Block gốc | Tắt |
| V7 | Chồng lấn nửa block, chọn cửa sổ có ngữ cảnh tốt nhất | Tắt |
| residual_only | Block gốc | Bật |
| V17 | Chồng lấn như V7 | Bật |

Sau khi ghép logit từng snippet về đúng thứ tự video, cộng
`eta × (2 × sigmoid(C) − 1)` vào mỗi logit A bất thường. Logit Normal giữ
nguyên và điểm A vẫn là `1 − softmax(A)[Normal]`. Công thức tương đương cộng
một độ lệch có dấu vào margin bất thường so với Normal. `eta = 0.10` là
giả định thăm dò khóa trước, chưa hiệu chuẩn. Tất cả cấu hình dùng checkpoint
gốc để suy luận. Neo Normal của V16 được đánh giá riêng.

## Input Kaggle

1. UCF test 290/crop 5, `ucfclipfeatures_test290.zip` hoặc cây feature đã giải nén.
2. Checkpoint UCF V0, `model_ucf.pth` hoặc ZIP checkpoint đã dùng trong V16.
3. **VadCLIP XD-Violence Test 800**, cùng feature và manifest đã dùng trong V16.
4. Checkpoint XD V0, `model_xd.pth` hoặc ZIP checkpoint đã dùng trong V16.

Gắn lại bốn input của V16. Bật Internet để tải mã ở commit đã khóa và CLIP
ViT-B/16, chọn GPU T4. Có thể đặt đường dẫn ở cell cấu hình nếu Kaggle có nhiều input trùng tên.

## Chạy và đọc kết quả

1. Import notebook, giữ `RUN_MODE = "smoke"`, Run All. Cả ucf và xd phải báo
   `ok`. Smoke chạy cả bốn cấu hình, gồm video dài hơn một block.
2. Đổi `RUN_MODE = "full"`, Run All. Dự trù 15–40 phút trên T4 với input sẵn,
   tùy thời gian đọc feature và tải nguồn. Đây là ước lượng, chưa đo V17 trên Kaggle.
3. Tải ZIP nhẹ ở `/kaggle/working/vadclip_v17/run_*/vadclip_v17_results.zip`.
   ZIP gồm bảng so sánh, log, metadata và mã chạy. Bật
   `INCLUDE_PREDICTIONS = True` nếu muốn kèm điểm snippet để replay.

Dòng `v17_vs_v7` đo riêng residual khi đã dùng cửa sổ V7. Dòng `v17_vs_v0` đo
hiệu ứng của toàn bộ tổ hợp. Dòng `residual_only_vs_v0` đo residual với block gốc.
Notebook kiểm dự đoán C của V17 khớp V7 và C của residual_only khớp V0.
UCF dùng ROC-AUC nhánh C làm chỉ số chính. XD dùng frame AP nhánh A. Đọc thêm
ROC-AUC/AP nhánh A của UCF để đo tác động của residual. Khoảng tin cậy dùng
500 bootstrap ghép cặp theo video. Đây là đánh giá thăm dò trên test đã xem.

Cuối lượt full, log in bảng điểm và bảng chênh lệch kèm khoảng tin cậy.
ZIP có `comparison.md` để đọc bảng và `comparison.csv` để lọc mọi đối chiếu.
