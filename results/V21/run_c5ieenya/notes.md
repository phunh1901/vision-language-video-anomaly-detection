# V21 — run_c5ieenya

ZIP gốc: `results (7).zip`; lưu tại `result.zip`.
SHA-256: `063e9e876fd75310cd43891f6068258799d37124316844dd65b9c93427697624`.
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
| ucf | cosine_control | classification | 0.848882 | 0.242883 | 0.624769 |
| ucf | cosine_control | alignment | 0.844671 | 0.256629 | 0.607159 |
| ucf | v0 | classification | 0.880199 | 0.335573 | 0.702323 |
| ucf | v0 | alignment | 0.856938 | 0.265006 | 0.655220 |
| xd | cosine_control | branch_1 | 0.927173 | 0.722735 | N/A |
| xd | cosine_control | branch_2 | 0.913881 | 0.644663 | N/A |
| xd | v0 | branch_1 | 0.939089 | 0.803085 | N/A |
| xd | v0 | branch_2 | 0.946813 | 0.845045 | N/A |

Nguồn member:

- `result.zip:vadclip_v21/run_c5ieenya/ucf/cosine_control/full/run.json`
- `result.zip:vadclip_v21/run_c5ieenya/ucf/cosine_control/full/run.json`
- `result.zip:vadclip_v21/run_c5ieenya/ucf/v0/full/run.json`
- `result.zip:vadclip_v21/run_c5ieenya/ucf/v0/full/run.json`
- `result.zip:vadclip_v21/run_c5ieenya/xd/cosine_control/full/run_u518wreb/run.json`
- `result.zip:vadclip_v21/run_c5ieenya/xd/cosine_control/full/run_u518wreb/run.json`
- `result.zip:vadclip_v21/run_c5ieenya/xd/v0/full/run_jrrg4tul/run.json`
- `result.zip:vadclip_v21/run_c5ieenya/xd/v0/full/run_jrrg4tul/run.json`

## Kinh nghiệm và giới hạn khi đọc

- So primary UCF C ROC-AUC / XD A AP với V0 cùng run; nhánh phụ báo riêng.
- Lỗi/giảm metric vẫn là kết quả cần giữ; không gọi đọc JSON là kiểm chứng huấn luyện hoặc replay.
- Chỉ dùng source/config gắn với artifact, không mặc định notebook hiện tại là source của run cũ.
- Ngưỡng, latency và chế độ smoke/full phải đọc trong config/log; không suy từ tên ZIP.

## Báo cáo phân tích và kinh nghiệm đã có

- `analysis.zip:ANALYSIS_run_c5ieenya.md`
- `analysis.zip:ucf/error_analysis/ERROR_ANALYSIS.md`
- `analysis.zip:xd/error_analysis/ERROR_ANALYSIS.md`
- `result.zip:vadclip_v21/run_c5ieenya/ucf/error_analysis/ERROR_ANALYSIS.md`
- `result.zip:vadclip_v21/run_c5ieenya/xd/error_analysis/ERROR_ANALYSIS.md`

Trích phần mở đầu báo cáo đã lưu (không phải phân tích mới):

> # V21 ZIP7 — phân tích run_c5ieenya
>
> Ngày phân tích: 22/11/2026. Nguồn: `results (7).zip`, run full mới của người dùng,
> native cosine, LR UCF1e-5/XD2e-5, Adam, seed42, batch64, FP32,30epoch.
> Không huấn luyện hoặc sửa prediction trong lần phân tích này.
>
> **Kết luận:** hạ LR đã sửa sự cố tối ưu nghiêm trọng của nhánh C, nhưng chưa
> khôi phục recipe đạt V0. Phần tụt lớn nhất còn lại là nhánh alignment của XD:
> điểm số giữa các video không còn xếp hạng sự kiện/nền tốt, đặc biệt ở vùng
> tự tin cao. Đây là kết quả quan sát từ raw outputs; chưa chứng minh thành phần
> adapter, visual prompt hay optimizer nào là nguyên nhân đầu tiên.
>
> ## 1. Run đã hoàn tất, metric và checkpoint hợp lệ
>
> - Summary: UCF và XD đều `status=ok`, `postprocessing_errors=[]`; không dùng recovery.
> - ZIP CRC không lỗi;17/17 hash source thực thi khớp experiment manifest.
> - Tính lại AUC/AP của cả8 nhánh/variant từ prediction export khớp report, sai số <1e-12.
> - Hai best.pth đúng SHA đã khai báo, epoch30, tensor downstream hữu hạn, bằng đúng

## Output trên GitHub

[metrics.json](metrics.json) giữ metric gốc đọc từ JSON trong archive. ZIP, checkpoint và raw prediction giữ local; đọc JSON không thay replay độc lập.
