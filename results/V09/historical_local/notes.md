# V09 — historical_local

Collection local, không phải run_id hoặc ZIP tải nguyên gốc. `extra.zip` giữ
1 file độc nhất từ kết quả đã giải nén, có manifest SHA-256 bên trong.
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

## Output trên GitHub

[metrics.json](metrics.json) giữ metric gốc đọc từ JSON trong archive. ZIP, checkpoint và raw prediction giữ local; đọc JSON không thay replay độc lập.
