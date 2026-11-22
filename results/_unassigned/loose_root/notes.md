# Artifact gốc chưa gán version

extra.zip giữ run.json và predictions.npz từng nằm ở root kaggle_result,
không ở thư mục version. Đóng gói local kèm manifest/hash; chưa gán run/version
khi chưa đối chiếu nguồn. Không coi đây là ZIP tải nguyên gốc hoặc kết quả mới.

## Metric đọc từ artifact

Giá trị gốc dạng 0–1, giữ theo JSON. Đây là đọc báo cáo, chưa replay từ prediction.

| Dataset | Variant theo path | Nhánh | ROC-AUC | AP | Ano-AUC |
|---|---|---|---:|---:|---:|
| unknown | unknown | classification | 0.880199 | 0.335573 | 0.702323 |
| unknown | unknown | alignment | 0.856938 | 0.265006 | 0.655220 |

Nguồn member:

- `extra.zip:runs/imported_layout/run.json`
- `extra.zip:runs/imported_layout/run.json`

## Kinh nghiệm và giới hạn khi đọc

- So primary UCF C ROC-AUC / XD A AP với V0 cùng run; nhánh phụ báo riêng.
- Lỗi/giảm metric vẫn là kết quả cần giữ; không gọi đọc JSON là kiểm chứng huấn luyện hoặc replay.
- Chỉ dùng source/config gắn với artifact, không mặc định notebook hiện tại là source của run cũ.
- Ngưỡng, latency và chế độ smoke/full phải đọc trong config/log; không suy từ tên ZIP.

## Output trên GitHub

[metrics.json](metrics.json) giữ metric gốc đọc từ JSON trong archive. ZIP, checkpoint và raw prediction giữ local; đọc JSON không thay replay độc lập.
