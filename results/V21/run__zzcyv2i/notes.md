# V21 — run__zzcyv2i

ZIP gốc: `results (1).zip`; lưu tại `result.zip`.
SHA-256: `284ab8a92305fde0851a7e9382e1f132dc910f0a6bba42354bd617a23265e50a`.
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
| ucf | v21 | classification | 0.845679 | 0.221887 | 0.623144 |
| ucf | v21 | alignment | 0.833754 | 0.227581 | 0.576783 |
| xd | v0 | branch_1 | 0.939089 | 0.803085 | N/A |
| xd | v0 | branch_2 | 0.946813 | 0.845045 | N/A |
| xd | v21 | branch_1 | 0.851311 | 0.507244 | N/A |
| xd | v21 | branch_2 | 0.922705 | 0.768522 | N/A |

Nguồn member:

- `result.zip:vadclip_v21/run__zzcyv2i/ucf/v0/full/run.json`
- `result.zip:vadclip_v21/run__zzcyv2i/ucf/v0/full/run.json`
- `result.zip:vadclip_v21/run__zzcyv2i/ucf/v21/full/run.json`
- `result.zip:vadclip_v21/run__zzcyv2i/ucf/v21/full/run.json`
- `result.zip:vadclip_v21/run__zzcyv2i/xd/v0/full/run_q2712pwb/run.json`
- `result.zip:vadclip_v21/run__zzcyv2i/xd/v0/full/run_q2712pwb/run.json`
- `result.zip:vadclip_v21/run__zzcyv2i/xd/v21/full/run_oibq1elc/run.json`
- `result.zip:vadclip_v21/run__zzcyv2i/xd/v21/full/run_oibq1elc/run.json`

## Kinh nghiệm và giới hạn khi đọc

- So primary UCF C ROC-AUC / XD A AP với V0 cùng run; nhánh phụ báo riêng.
- Lỗi/giảm metric vẫn là kết quả cần giữ; không gọi đọc JSON là kiểm chứng huấn luyện hoặc replay.
- Chỉ dùng source/config gắn với artifact, không mặc định notebook hiện tại là source của run cũ.
- Ngưỡng, latency và chế độ smoke/full phải đọc trong config/log; không suy từ tên ZIP.

## Output trên GitHub

[metrics.json](metrics.json) giữ metric gốc đọc từ JSON trong archive. ZIP, checkpoint và raw prediction giữ local; đọc JSON không thay replay độc lập.
