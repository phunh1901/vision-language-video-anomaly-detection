# V21 — historical_local

Collection local, không phải run_id hoặc ZIP tải nguyên gốc. `extra.zip` giữ
52 file độc nhất từ kết quả đã giải nén, có manifest SHA-256 bên trong.
Phần trùng đã được đối chiếu SHA-256 với member trong ZIP gốc trước khi bỏ bản dư.
Collection có thể chứa nhiều run hoặc lần phân tích; không tự suy thời gian từ tên folder.

Giữ log, JSON/CSV, prediction và báo cáo phân tích để đọc lại kết quả/kinh nghiệm.
Các kết luận trước đây nằm trong file phân tích gốc và nhật ký; chưa replay toàn bộ
trong tác vụ tái tổ chức.

## Metric đọc từ artifact

Chưa tìm thấy run.json có schema metric tương ứng trong gói này. Không điền 0 hoặc suy kết quả.

## Kinh nghiệm và giới hạn khi đọc

- So primary UCF C ROC-AUC / XD A AP với V0 cùng run; nhánh phụ báo riêng.
- Lỗi/giảm metric vẫn là kết quả cần giữ; không gọi đọc JSON là kiểm chứng huấn luyện hoặc replay.
- Chỉ dùng source/config gắn với artifact, không mặc định notebook hiện tại là source của run cũ.
- Ngưỡng, latency và chế độ smoke/full phải đọc trong config/log; không suy từ tên ZIP.

## Báo cáo phân tích và kinh nghiệm đã có

- `extra.zip:ANALYSIS_run_4jua44k6.md`
- `extra.zip:ANALYSIS_run_4__dwo_a.md`
- `extra.zip:ANALYSIS_run_yxfrilug.md`
- `extra.zip:audit_run_yxfrilug/ucf/error_analysis/ERROR_ANALYSIS.md`
- `extra.zip:audit_run_yxfrilug/xd/error_analysis/ERROR_ANALYSIS.md`

Trích phần mở đầu báo cáo đã lưu (không phải phân tích mới):

> # Phân tích V21 — results (5).zip / run_4jua44k6
>
> Ngày kiểm tra: 21/11/2026. Phân tích kỹ thuật local, không phải nội dung viết thay
> quyển đồ án. Đọc trực tiếp ZIP gốc, log train, source thực thi, hai checkpoint cuối,
> optimizer state, history, learning health và predictions của cả hai dataset.
>
> **Kết luận:** run đã thực thi đúng bản SANITY_FIXED_C. Những sửa số học đã có tác
> dụng, nhưng chưa khắc phục nghiệm học phân biệt video thay vì định vị sự kiện.
> Phát hiện mới mạnh nhất là saturation của phép nén tangent ở visual; entailment
> có gradient nhưng quá yếu so với alignment và chưa thỏa cone ở các cặp được log.
> Đây là bằng chứng về cơ chế thất bại, chưa phải ablation định lượng đóng góp của
> từng nguyên nhân vào mức giảm AUC/AP.
>
> ## 1. Xác minh đúng lượt chạy và độ tin cậy đầu ra
>
> - ZIP: `results (5).zip`, SHA256
>   `53fcf192069f1ff90dbdd3ee4f5b4e2a32db1fef1d641112ed8486d65819ff8a`.
>   Kiểm CRC toàn ZIP đạt.

## Output trên GitHub

[metrics.json](metrics.json) giữ metric gốc đọc từ JSON trong archive. ZIP, checkpoint và raw prediction giữ local; đọc JSON không thay replay độc lập.
