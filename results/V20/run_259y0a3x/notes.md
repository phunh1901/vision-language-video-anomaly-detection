# V20 — run_259y0a3x

ZIP gốc: `vadclip_v20_essential_run_259y0a3x.zip`; lưu tại `result.zip`.
SHA-256: `1fb6d0d2f3334aef615bc23b3fdec226d742322f3241a346303e874249b14182`.
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
| ucf | normal_only | classification | 0.879800 | 0.334895 | 0.701202 |
| ucf | normal_only | alignment | 0.857576 | 0.265539 | 0.656351 |
| ucf | v0 | classification | 0.880199 | 0.335573 | 0.702323 |
| ucf | v0 | alignment | 0.856938 | 0.265006 | 0.655220 |
| ucf | v20 | classification | 0.880473 | 0.336205 | 0.703002 |
| ucf | v20 | alignment | 0.857576 | 0.265539 | 0.656351 |
| ucf | v7 | classification | 0.880234 | 0.335194 | 0.702324 |
| ucf | v7 | alignment | 0.857576 | 0.265539 | 0.656351 |
| xd | normal_only | branch_1 | 0.939944 | 0.807801 | N/A |
| xd | normal_only | branch_2 | 0.948155 | 0.848598 | N/A |
| xd | v0 | branch_1 | 0.939089 | 0.803085 | N/A |
| xd | v0 | branch_2 | 0.946813 | 0.845045 | N/A |
| xd | v20 | branch_1 | 0.939944 | 0.807801 | N/A |
| xd | v20 | branch_2 | 0.948133 | 0.848166 | N/A |
| xd | v7 | branch_1 | 0.939944 | 0.807801 | N/A |
| xd | v7 | branch_2 | 0.947442 | 0.846198 | N/A |

Nguồn member:

- `result.zip:ucf/normal_only/full/run.json`
- `result.zip:ucf/normal_only/full/run.json`
- `result.zip:ucf/v0/full/run.json`
- `result.zip:ucf/v0/full/run.json`
- `result.zip:ucf/v20/full/run.json`
- `result.zip:ucf/v20/full/run.json`
- `result.zip:ucf/v7/full/run.json`
- `result.zip:ucf/v7/full/run.json`
- `result.zip:xd/normal_only/full/run_zgthiqx0/run.json`
- `result.zip:xd/normal_only/full/run_zgthiqx0/run.json`
- `result.zip:xd/v0/full/run_t0d3syyc/run.json`
- `result.zip:xd/v0/full/run_t0d3syyc/run.json`
- `result.zip:xd/v20/full/run_7e7x50r5/run.json`
- `result.zip:xd/v20/full/run_7e7x50r5/run.json`
- `result.zip:xd/v7/full/run_x93t3byh/run.json`
- `result.zip:xd/v7/full/run_x93t3byh/run.json`

## Kinh nghiệm và giới hạn khi đọc

- So primary UCF C ROC-AUC / XD A AP với V0 cùng run; nhánh phụ báo riêng.
- Lỗi/giảm metric vẫn là kết quả cần giữ; không gọi đọc JSON là kiểm chứng huấn luyện hoặc replay.
- Chỉ dùng source/config gắn với artifact, không mặc định notebook hiện tại là source của run cũ.
- Ngưỡng, latency và chế độ smoke/full phải đọc trong config/log; không suy từ tên ZIP.

## Output trên GitHub

[metrics.json](metrics.json) giữ metric gốc đọc từ JSON trong archive. ZIP, checkpoint và raw prediction giữ local; đọc JSON không thay replay độc lập.
