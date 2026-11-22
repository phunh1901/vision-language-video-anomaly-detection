# V02 — historical_local

Collection local, không phải run_id hoặc ZIP tải nguyên gốc. `extra.zip` giữ
6 file độc nhất từ kết quả đã giải nén, có manifest SHA-256 bên trong.
Phần trùng đã được đối chiếu SHA-256 với member trong ZIP gốc trước khi bỏ bản dư.
Collection có thể chứa nhiều run hoặc lần phân tích; không tự suy thời gian từ tên folder.

Giữ log, JSON/CSV, prediction và báo cáo phân tích để đọc lại kết quả/kinh nghiệm.
Các kết luận trước đây nằm trong file phân tích gốc và nhật ký; chưa replay toàn bộ
trong tác vụ tái tổ chức.

## Metric đọc từ artifact

Giá trị gốc dạng 0–1, giữ theo JSON. Đây là đọc báo cáo, chưa replay từ prediction.

| Dataset | Variant theo path | Nhánh | ROC-AUC | AP | Ano-AUC |
|---|---|---|---:|---:|---:|
| ucf | không ghi trong member path | classification | 0.880199 | 0.335573 | 0.702323 |
| ucf | không ghi trong member path | alignment | 0.856938 | 0.265006 | 0.655220 |
| ucf | không ghi trong member path | positive_only | 0.859556 | 0.257058 | 0.663206 |
| ucf | không ghi trong member path | contrast_v2 | 0.827780 | 0.199219 | 0.612587 |
| xd | không ghi trong member path | branch_1 | 0.939089 | N/A | N/A |
| xd | không ghi trong member path | branch_2 | 0.946813 | N/A | N/A |
| xd | không ghi trong member path | positive_only | 0.835654 | N/A | N/A |
| xd | không ghi trong member path | contrast_v2 | 0.833485 | N/A | N/A |

Nguồn member:

- `extra.zip:UCF/run.json`
- `extra.zip:UCF/run.json`
- `extra.zip:UCF/run.json`
- `extra.zip:UCF/run.json`
- `extra.zip:XD/run.json`
- `extra.zip:XD/run.json`
- `extra.zip:XD/run.json`
- `extra.zip:XD/run.json`

## Kinh nghiệm và giới hạn khi đọc

- So primary UCF C ROC-AUC / XD A AP với V0 cùng run; nhánh phụ báo riêng.
- Lỗi/giảm metric vẫn là kết quả cần giữ; không gọi đọc JSON là kiểm chứng huấn luyện hoặc replay.
- Chỉ dùng source/config gắn với artifact, không mặc định notebook hiện tại là source của run cũ.
- Ngưỡng, latency và chế độ smoke/full phải đọc trong config/log; không suy từ tên ZIP.

## Output trên GitHub

[metrics.json](metrics.json) giữ metric gốc đọc từ JSON trong archive. ZIP, checkpoint và raw prediction giữ local; đọc JSON không thay replay độc lập.
