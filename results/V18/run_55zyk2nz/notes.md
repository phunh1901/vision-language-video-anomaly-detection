# V18 — run_55zyk2nz

ZIP gốc: `vadclip_v18_essential_run_55zyk2nz.zip`; lưu tại `result.zip`.
SHA-256: `1210fe7a5e5eb39d93505e0b29fa776fc01c9c342b64aa709af0720f662f0073`.
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
| ucf | clip_only | classification | 0.880234 | 0.335194 | 0.702324 |
| ucf | clip_only | alignment | 0.857576 | 0.265539 | 0.656351 |
| ucf | temporal_random | classification | 0.880234 | 0.335194 | 0.702324 |
| ucf | temporal_random | alignment | 0.857576 | 0.265539 | 0.656351 |
| ucf | v18 | classification | 0.880234 | 0.335194 | 0.702324 |
| ucf | v18 | alignment | 0.857576 | 0.265539 | 0.656351 |
| ucf | v7 | classification | 0.880234 | 0.335194 | 0.702324 |
| ucf | v7 | alignment | 0.857576 | 0.265539 | 0.656351 |
| xd | clip_only | branch_1 | 0.939944 | 0.807801 | N/A |
| xd | clip_only | branch_2 | 0.943264 | 0.822674 | N/A |
| xd | temporal_random | branch_1 | 0.939944 | 0.807801 | N/A |
| xd | temporal_random | branch_2 | 0.947021 | 0.836797 | N/A |
| xd | v18 | branch_1 | 0.939944 | 0.807801 | N/A |
| xd | v18 | branch_2 | 0.941940 | 0.797592 | N/A |
| xd | v7 | branch_1 | 0.939944 | 0.807801 | N/A |
| xd | v7 | branch_2 | 0.947442 | 0.846198 | N/A |

Nguồn member:

- `result.zip:ucf/clip_only/full/run.json`
- `result.zip:ucf/clip_only/full/run.json`
- `result.zip:ucf/temporal_random/full/run.json`
- `result.zip:ucf/temporal_random/full/run.json`
- `result.zip:ucf/v18/full/run.json`
- `result.zip:ucf/v18/full/run.json`
- `result.zip:ucf/v7/full/run.json`
- `result.zip:ucf/v7/full/run.json`
- `result.zip:xd/clip_only/full/run_67a827d7/run.json`
- `result.zip:xd/clip_only/full/run_67a827d7/run.json`
- `result.zip:xd/temporal_random/full/run_mb2edhph/run.json`
- `result.zip:xd/temporal_random/full/run_mb2edhph/run.json`
- `result.zip:xd/v18/full/run_e9a_6vsx/run.json`
- `result.zip:xd/v18/full/run_e9a_6vsx/run.json`
- `result.zip:xd/v7/full/run_xezgb20t/run.json`
- `result.zip:xd/v7/full/run_xezgb20t/run.json`

## Kinh nghiệm và giới hạn khi đọc

- So primary UCF C ROC-AUC / XD A AP với V0 cùng run; nhánh phụ báo riêng.
- Lỗi/giảm metric vẫn là kết quả cần giữ; không gọi đọc JSON là kiểm chứng huấn luyện hoặc replay.
- Chỉ dùng source/config gắn với artifact, không mặc định notebook hiện tại là source của run cũ.
- Ngưỡng, latency và chế độ smoke/full phải đọc trong config/log; không suy từ tên ZIP.

## Output trên GitHub

[metrics.json](metrics.json) giữ metric gốc đọc từ JSON trong archive. ZIP, checkpoint và raw prediction giữ local; đọc JSON không thay replay độc lập.
