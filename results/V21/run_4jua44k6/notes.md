# V21 — run_4jua44k6

ZIP gốc: `results (5).zip`; lưu tại `result.zip`.
SHA-256: `53fcf192069f1ff90dbdd3ee4f5b4e2a32db1fef1d641112ed8486d65819ff8a`.
Định danh run lấy từ tên ZIP hoặc prefix trong archive; các ID con/ID tham chiếu
khác trong metadata không tự coi là run chính. Ngày chạy: chưa xác minh đủ timestamp.
Ngày tổ chức: 23/11/2026; không phải ngày chạy/công bố.

## Nguồn, kết quả và kinh nghiệm

Source/config/raw logs/metrics gốc giữ trong ZIP. Việc chuyển thư mục chưa replay
metric và không sinh kết quả mới. Xem tổng hợp kinh nghiệm ở notes của collection
cùng version và các ghi chú kết quả tương ứng.

Giữ kết quả âm và lỗi; không dùng test để chọn lại threshold/config. Nếu ZIP nhẹ
không chứa prediction thì không tự tuyên bố metric đã replay được.

## Metric đọc từ artifact

Giá trị gốc dạng 0–1, giữ theo JSON. Đây là đọc báo cáo, chưa replay từ prediction.

| Dataset | Variant theo path | Nhánh | ROC-AUC | AP | Ano-AUC |
|---|---|---|---:|---:|---:|
| ucf | v0 | classification | 0.880199 | 0.335573 | 0.702323 |
| ucf | v0 | alignment | 0.856938 | 0.265006 | 0.655220 |
| ucf | v21 | classification | 0.800597 | 0.198726 | 0.570506 |
| ucf | v21 | alignment | 0.719761 | 0.140523 | 0.523921 |
| xd | v0 | branch_1 | 0.939089 | 0.803085 | N/A |
| xd | v0 | branch_2 | 0.946813 | 0.845045 | N/A |
| xd | v21 | branch_1 | 0.876928 | 0.569621 | N/A |
| xd | v21 | branch_2 | 0.864867 | 0.496146 | N/A |

Nguồn member:

- `result.zip:vadclip_v21/run_4jua44k6/ucf/v0/full/run.json`
- `result.zip:vadclip_v21/run_4jua44k6/ucf/v0/full/run.json`
- `result.zip:vadclip_v21/run_4jua44k6/ucf/v21/full/run.json`
- `result.zip:vadclip_v21/run_4jua44k6/ucf/v21/full/run.json`
- `result.zip:vadclip_v21/run_4jua44k6/xd/v0/full/run_7_lcbcdc/run.json`
- `result.zip:vadclip_v21/run_4jua44k6/xd/v0/full/run_7_lcbcdc/run.json`
- `result.zip:vadclip_v21/run_4jua44k6/xd/v21/full/run_qi0zy67k/run.json`
- `result.zip:vadclip_v21/run_4jua44k6/xd/v21/full/run_qi0zy67k/run.json`

## Kinh nghiệm và giới hạn khi đọc

- So primary UCF C ROC-AUC / XD A AP với V0 cùng run; nhánh phụ báo riêng.
- Lỗi/giảm metric vẫn là kết quả cần giữ; không gọi đọc JSON là kiểm chứng huấn luyện hoặc replay.
- Chỉ dùng source/config gắn với artifact, không mặc định notebook hiện tại là source của run cũ.
- Ngưỡng, latency và chế độ smoke/full phải đọc trong config/log; không suy từ tên ZIP.

## Output trên GitHub

[metrics.json](metrics.json) giữ metric gốc đọc từ JSON trong archive. ZIP, checkpoint và raw prediction giữ local; đọc JSON không thay replay độc lập.
