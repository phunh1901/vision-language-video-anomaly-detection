# VadCLIP V19 · Học phân biệt sự kiện và nền trong cùng video

Giữ checkpoint VadCLIP và cửa sổ chồng lấn V7. Huấn luyện cùng một head CLIP + temporal theo hai cách ghép mẫu âm. Cả hai biến thể dự đoán trên cùng official test với V7.

| Cấu hình | Huấn luyện |
|---|---|
| V7 | Mốc đối chứng |
| Temporal (Normal) | Mẫu âm từ video Normal |
| V19 | Thay một phần mẫu âm bằng đoạn nền tin cậy trong cùng video với mẫu dương |

V19 chọn đoạn dương theo điểm hai nhánh V7 và hỗ trợ lân cận. Đoạn nền phải có hai điểm thấp liên tục trong cửa sổ 9 snippet, cách đoạn dương 5–64 snippet và có cosine CLIP từ 0,45. Mỗi video thay tối đa một nửa số cặp, làm tròn lên. Hai head dùng cùng lịch batch, trọng số cặp và khởi tạo. Loss học xếp hạng đầu ra head, kết hợp MIL trên điểm V7 đã hiệu chỉnh. Gate dùng mức bằng chứng V7 có hỗ trợ lân cận. Hai head dùng chung trần hiệu chỉnh 0,25 logit khóa trước. Video AP development dùng để theo dõi. Các đoạn được khai thác là nhãn giả.

## Sáu input Kaggle

- VadCLIP UCF-Crime Test290
- VadCLIP UCF Checkpoint
- VadCLIP XD-Violence Test 800
- VadCLIP XD-Violence Checkpoint
- VadCLIP UCF-Crime Train1610 V5
- VadCLIP XD-Violence Train3954 V5

## Cách chạy

1. Gắn sáu input, bật Internet và chọn GPU T4.
2. Giữ RUN_MODE = "smoke" rồi Run All. Smoke chạy hai batch mỗi tập và kiểm tra luồng lưu checkpoint, nạp lại và chấm. Mẫu smoke có thể thiếu cặp nền hợp lệ.
3. Khi hai tập báo ok, đổi RUN_MODE = "full" và Run All. Full yêu cầu có cặp thay thế trên mỗi tập trước khi chấm hai head. Ước tính sơ bộ 15–25 phút trên T4 dựa trên V18, chưa đo trên Kaggle.
4. Tải vadclip_v19_results.zip trong panel Output theo đường dẫn Download cuối log. ZIP mặc định lưu bảng so sánh, summary, log, checkpoint và mined_pairs.json. Bật INCLUDE_PREDICTIONS để lưu điểm phục vụ chấm lại.

Chỉ số chính là frame ROC-AUC (UCF, C) và frame AP (XD, A). So V19 với Temporal (Normal) để đánh giá việc thay mẫu âm. Summary ghi FPR nền trong video bất thường tại recall 50% và 80%, recall theo độ dài sự kiện và số snippet đổi điểm. Các ngưỡng dựa trên nhãn official test phục vụ phân tích hồi cứu.
