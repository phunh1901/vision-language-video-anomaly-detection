# V16 — run_eljfkmkf

ZIP gốc: `vadclip_v16_essential_run_eljfkmkf.zip`; lưu tại `result.zip`.
SHA-256: `b2419bdf9a9cbe3f0e2ab0563c4c2af6e126974bfc7c67b9880a89314421dfe0`.
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
| ucf | identity | classification | 0.880199 | 0.335573 | 0.702323 |
| ucf | identity | alignment | 0.856938 | 0.265006 | 0.655220 |
| ucf | v0 | classification | 0.880199 | 0.335573 | 0.702323 |
| ucf | v0 | alignment | 0.856938 | 0.265006 | 0.655220 |
| ucf | v16 | classification | 0.880199 | 0.335573 | 0.702323 |
| ucf | v16 | alignment | 0.858729 | 0.269116 | 0.659375 |
| xd | identity | branch_1 | 0.939089 | 0.803085 | N/A |
| xd | identity | branch_2 | 0.946813 | 0.845045 | N/A |
| xd | v0 | branch_1 | 0.939089 | 0.803085 | N/A |
| xd | v0 | branch_2 | 0.946813 | 0.845045 | N/A |
| xd | v16 | branch_1 | 0.939089 | 0.803085 | N/A |
| xd | v16 | branch_2 | 0.944933 | 0.839940 | N/A |

Nguồn member:

- `result.zip:ucf/identity/full/run.json`
- `result.zip:ucf/identity/full/run.json`
- `result.zip:ucf/v0/full/run.json`
- `result.zip:ucf/v0/full/run.json`
- `result.zip:ucf/v16/full/run.json`
- `result.zip:ucf/v16/full/run.json`
- `result.zip:xd/identity/full/run_8b9a7lst/run.json`
- `result.zip:xd/identity/full/run_8b9a7lst/run.json`
- `result.zip:xd/v0/full/run_o8oyfxf4/run.json`
- `result.zip:xd/v0/full/run_o8oyfxf4/run.json`
- `result.zip:xd/v16/full/run_gasbyf4v/run.json`
- `result.zip:xd/v16/full/run_gasbyf4v/run.json`

## Kinh nghiệm và giới hạn khi đọc

- So primary UCF C ROC-AUC / XD A AP với V0 cùng run; nhánh phụ báo riêng.
- Lỗi/giảm metric vẫn là kết quả cần giữ; không gọi đọc JSON là kiểm chứng huấn luyện hoặc replay.
- Chỉ dùng source/config gắn với artifact, không mặc định notebook hiện tại là source của run cũ.
- Ngưỡng, latency và chế độ smoke/full phải đọc trong config/log; không suy từ tên ZIP.

## Output trên GitHub

[metrics.json](metrics.json) giữ metric gốc đọc từ JSON trong archive. ZIP, checkpoint và raw prediction giữ local; đọc JSON không thay replay độc lập.
