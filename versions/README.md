# Code chạy các version

Chọn notebook trong bảng, import lên Kaggle và đọc cell config trước Run All.

| Version | Notebook chuẩn | Mô tả từ notebook |
|---|---|---|
| V00 | [V00.ipynb](V00.ipynb) | VadCLIP V0 · checkpoint gốc trên UCF-Crime và XD-Violence |
| V01 | [V01.ipynb](V01.ipynb) | VadCLIP V1 trên UCF-Crime và XD-Violence |
| V02 | [V02.ipynb](V02.ipynb) | VadCLIP V2/H3 — Prompt đối lập trên UCF-Crime và XD-Violence |
| V03 | [V03.ipynb](V03.ipynb) | VadCLIP V3 — Lọc đỉnh điểm ngắn trên UCF-Crime và XD-Violence |
| V04 | [V04.ipynb](V04.ipynb) | VadCLIP V4 — Kết hợp hai nhánh có điều kiện |
| V05 | [V05.ipynb](V05.ipynb) | VadCLIP V5 — Học xếp hạng với video Normal khó |
| V06 | [V06.ipynb](V06.ipynb) | VadCLIP V6 — Adapter thời gian đa tỉ lệ |
| V07 | [V07.ipynb](V07.ipynb) | VadCLIP V7 — Suy luận với cửa sổ chồng lấn |
| V08 | [V08.ipynb](V08.ipynb) | VadCLIP V8 — Học bao phủ sự kiện bằng nhãn mềm cục bộ |
| V09 | [V09.ipynb](V09.ipynb) | VadCLIP V9 — Loss bao phủ một chiều |
| V10 | [V10.ipynb](V10.ipynb) | VadCLIP V10 — Chọn top-k từ nhiều vùng thời gian |
| V11 | [V11.ipynb](V11.ipynb) | VadCLIP V11 — Học thứ tự điểm giữa hai đoạn trong cùng video |
| V12 | [V12.ipynb](V12.ipynb) | VadCLIP V12 — Tách mục tiêu ngắn hạn và dài hạn |
| V13 | [V13.ipynb](V13.ipynb) | VadCLIP V13 — Huấn luyện MIL trên các block liên tiếp |
| V14 | [V14.ipynb](V14.ipynb) | VadCLIP V14 — Hiệu chỉnh V0 theo diễn biến thời gian |
| V15 | [V15.ipynb](V15.ipynb) | VadCLIP V15 — Hoán đổi điểm trong từng video |
| V16 | [V16.ipynb](V16.ipynb) | V16 — Neo Normal thích ứng theo video (lộ trình mới, 14/11/2026) |
| V17 | [V17.ipynb](V17.ipynb) | VadCLIP V17 — Residual C→A và cửa sổ chồng lấn V7 |
| V18 | [V18.ipynb](V18.ipynb) | VadCLIP V18 · Biến thiên CLIP và mẫu âm khó trên nền V7 |
| V19 | [V19.ipynb](V19.ipynb) | VadCLIP V19 · Học phân biệt sự kiện và nền trong cùng video |
| V20 | [V20.ipynb](V20.ipynb) | VadCLIP V20 · Cân bằng hiệu chỉnh temporal theo video |
| V21 | [V21.ipynb](V21.ipynb) | V21 — notebook chính duy nhất |

V0–V15 đã có code nhúng trong notebook, không thiếu version vì không có folder Python riêng.
Source dùng sửa/build V16–V21 nằm trong [_dev](_dev/README.md).

## Input và cách chạy

- Test UCF: đủ 290 video, crop __5 theo CSV; XD: VadCLIP XD-Violence Test 800, crop __0.
- Checkpoint V0 riêng UCF/XD và CLIP khi đường chạy yêu cầu; notebook kiểm hash.
- V21 native cần train đủ 10 crop/video: UCF 16.100 file, XD 39.540 file. Không dùng thay bằng train_v5 một crop.
- [Guide V21 hiện hành](_dev/v21/NOTEBOOK_GUIDE_COSINE.md). Không cần output V7/V20 làm input.
- Kiểm RUN_MODE: một số notebook lưu full. Chạy smoke khi cần kiểm đường chạy trước.
- Dùng cell cuối tải ZIP; lưu vào results/Vxx/run_id và ghi notes. ZIP nhẹ có thể thiếu predictions.

Ngày tạo/chạy nghiên cứu cần nguồn riêng; tên file/mtime/ngày commit không chứng minh ngày chạy.
