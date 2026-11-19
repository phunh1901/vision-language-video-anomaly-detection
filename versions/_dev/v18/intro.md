# VadCLIP V18 · Biến thiên CLIP và mẫu âm khó trên nền V7

V18 kiểm tra liệu biến thiên embedding theo thời gian và học với mẫu âm có hình ảnh tương tự có giúp phân biệt hành vi bất thường trong bối cảnh dễ nhầm hay không. Nhánh hiệu chỉnh nhận feature CLIP hiện tại và chênh lệch có dấu với các snippet trước và sau, cách 1, 2 và 4 bước. Tín hiệu được tính trên từng video, bao gồm cả các cặp qua ranh giới block V7. Vị trí thiếu hàng xóm đóng góp 0.

| Cấu hình | Tín hiệu | Cách ghép mẫu âm |
|---|---|---|
| V7 | Checkpoint VadCLIP và cửa sổ chồng lấn | Đối chứng |
| CLIP | Feature hiện tại | Ngẫu nhiên |
| CLIP + temporal | Feature hiện tại và chênh lệch embedding | Ngẫu nhiên |
| V18 | Feature hiện tại và chênh lệch embedding | Độ tương đồng cosine cao nhất trong nhóm mẫu âm của batch |

Ba nhánh học dùng chung split theo nhóm nguồn, khởi tạo và lịch batch. Các đoạn dương được sàng lọc bằng teacher V7 và sự đồng thuận với snippet lân cận. Mẫu âm đến từ video Normal. Hệ số residual được chọn theo AP cấp video trên development. UCF hiệu chỉnh logit nhánh C, XD hiệu chỉnh logit Normal của nhánh A.

## Sáu input Kaggle

| Dataset | Dữ liệu |
|---|---|
| VadCLIP UCF-Crime Test290 | Feature CLIP test crop 5 |
| VadCLIP UCF Checkpoint | model_ucf.pth |
| VadCLIP XD-Violence Test 800 | Feature CLIP test và manifest gốc |
| VadCLIP XD-Violence Checkpoint | model_xd.pth |
| VadCLIP UCF-Crime Train1610 V5 | vadclip_ucf_train_v5.zip |
| VadCLIP XD-Violence Train3954 V5 | vadclip_xd_train_v5.zip |

## Cách chạy

1. Gắn sáu input, chọn GPU T4 và bật Internet để nạp mã VadCLIP và trọng số CLIP.
2. Giữ RUN_MODE = "smoke" và chọn Run All. Smoke chạy hai batch train mỗi tập và kiểm tra cả bốn cấu hình.
3. Khi hai tập báo ok, đặt RUN_MODE = "full" và chạy lại.
4. Tải /kaggle/working/vadclip_v18/run_*/vadclip_v18_results.zip. Gói gồm bảng comparison.md và comparison.csv ở chế độ full, summary.json, log và checkpoint residual. Bật INCLUDE_PREDICTIONS để lưu điểm theo snippet.

Đọc hàng V18 so với V7 cho frame ROC-AUC trên UCF-Crime và frame AP trên XD-Violence. So CLIP + temporal với CLIP để đo hiệu ứng biến thiên thời gian, rồi so V18 với CLIP + temporal để đo hiệu ứng mẫu âm khó. Xem thêm tỷ lệ báo động giả ở các mục tiêu recall 50% và 80%, recall thực tế sau khi xét điểm trùng, và recall sự kiện ở cùng trần FPR. Các chỉ số theo ngưỡng trên test dùng cho phân tích hồi cứu. Chênh lệch embedding có thể phản ứng với chuyển cảnh và chỉ phản ánh tín hiệu đã được giữ trong feature CLIP.
