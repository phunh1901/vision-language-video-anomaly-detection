# V19 — run_tran7l08

ZIP gốc: `vadclip_v19_essential_run_tran7l08.zip`; lưu tại `result.zip`.
SHA-256: `e9debfe1934f42a08412a9270a1a7924a918943be0b0829531b99263bce1d3c2`.
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
| ucf | v19 | classification | 0.879847 | 0.334960 | 0.701340 |
| ucf | v19 | alignment | 0.857576 | 0.265539 | 0.656351 |
| ucf | v7 | classification | 0.880234 | 0.335194 | 0.702324 |
| ucf | v7 | alignment | 0.857576 | 0.265539 | 0.656351 |
| xd | normal_only | branch_1 | 0.939944 | 0.807801 | N/A |
| xd | normal_only | branch_2 | 0.948155 | 0.848598 | N/A |
| xd | v19 | branch_1 | 0.939944 | 0.807801 | N/A |
| xd | v19 | branch_2 | 0.948209 | 0.848710 | N/A |
| xd | v7 | branch_1 | 0.939944 | 0.807801 | N/A |
| xd | v7 | branch_2 | 0.947442 | 0.846198 | N/A |

Nguồn member:

- `result.zip:ucf/normal_only/full/run.json`
- `result.zip:ucf/normal_only/full/run.json`
- `result.zip:ucf/v19/full/run.json`
- `result.zip:ucf/v19/full/run.json`
- `result.zip:ucf/v7/full/run.json`
- `result.zip:ucf/v7/full/run.json`
- `result.zip:xd/normal_only/full/run_4vih28db/run.json`
- `result.zip:xd/normal_only/full/run_4vih28db/run.json`
- `result.zip:xd/v19/full/run_th5rb9rz/run.json`
- `result.zip:xd/v19/full/run_th5rb9rz/run.json`
- `result.zip:xd/v7/full/run_70po75u1/run.json`
- `result.zip:xd/v7/full/run_70po75u1/run.json`

## Kinh nghiệm và giới hạn khi đọc

- So primary UCF C ROC-AUC / XD A AP với V0 cùng run; nhánh phụ báo riêng.
- Lỗi/giảm metric vẫn là kết quả cần giữ; không gọi đọc JSON là kiểm chứng huấn luyện hoặc replay.
- Chỉ dùng source/config gắn với artifact, không mặc định notebook hiện tại là source của run cũ.
- Ngưỡng, latency và chế độ smoke/full phải đọc trong config/log; không suy từ tên ZIP.

## Output trên GitHub

[metrics.json](metrics.json) giữ metric gốc đọc từ JSON trong archive. ZIP, checkpoint và raw prediction giữ local; đọc JSON không thay replay độc lập.
