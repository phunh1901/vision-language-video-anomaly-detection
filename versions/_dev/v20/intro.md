# VadCLIP V20 · Cân bằng hiệu chỉnh temporal theo video

V20 kiểm tra liệu loại bỏ thành phần hiệu chỉnh chung của từng video có giúp head temporal cải thiện thứ hạng frame giữa các video. V7 cho thêm ngữ cảnh tại biên block. Đối chứng Temporal (Normal) của V19 cho tín hiệu tốt trên XD khi giới hạn logit. V20 dùng đúng head và gate đó, rồi trừ trung bình đầu ra head có trọng số gate trước khi áp dụng hiệu chỉnh. Các snippet có gate bằng 0 giữ nguyên. Tổng hiệu chỉnh logit trên mỗi video xấp xỉ 0 và từng snippet giữ trần 0,25 logit. Tính chất này áp dụng trên logit, thứ hạng giữa video và xác suất sau sigmoid/softmax vẫn có thể thay đổi. V20 chạy offline vì cần điểm của toàn video.

| Cấu hình | Thành phần |
|---|---|
| V0 | Checkpoint gốc VadCLIP, block gốc |
| V7 | V0 với cửa sổ chồng lấn |
| V7 + temporal | Head đối chứng V19, mẫu âm từ video Normal |
| V20 | Cùng head và checkpoint với V7 + temporal, thêm cân bằng hiệu chỉnh theo video |

Head huấn luyện một lần mỗi tập bằng official train. Dùng chung cách chọn và trọng số mẫu của đối chứng V19 để so trực tiếp hai cách áp dụng head. AP cấp video trên development chỉ dùng để ghi nhận. Head hiệu chỉnh C trên UCF và logit Normal của A trên XD, giữ hướng điểm bất thường nhất quán.

## Sáu input Kaggle

- VadCLIP UCF-Crime Test290
- VadCLIP UCF Checkpoint
- VadCLIP XD-Violence Test 800
- VadCLIP XD-Violence Checkpoint
- VadCLIP UCF-Crime Train1610 V5
- VadCLIP XD-Violence Train3954 V5

## Cách chạy

1. Import notebook, gắn sáu input, chọn GPU T4 và bật Internet.
2. Giữ RUN_MODE = "smoke" và Run All. Kiểm tra cả hai tập báo ok, rồi đổi thành "full" và Run All. Dự kiến 15–25 phút theo lượt V19, chưa đo V20 trên Kaggle.
3. Tải vadclip_v20_results.zip từ panel Output theo đường dẫn cuối log. ZIP gồm bảng so sánh, summary, log và checkpoint temporal dùng chung. Bật INCLUDE_PREDICTIONS để lưu điểm phục vụ tính lại.

So sánh chính là V20 với V0 theo frame ROC-AUC UCF và frame AP XD. V7 với V0 đo cửa sổ chồng lấn, temporal với V7 đo head, V20 với temporal đo riêng phép cân bằng. Summary có FPR ở recall 50% và 80%, recall sự kiện ở cùng trần FPR và số snippet đổi so với V7. Các ngưỡng lấy từ nhãn test phục vụ chẩn đoán hồi cứu.
