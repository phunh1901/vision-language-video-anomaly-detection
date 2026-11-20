# V21 Sanity Fix — cách chạy hiện hành

Cập nhật 21/11/2026. **Bước chạy tiếp theo là đối chứng trong
`run_vadclip_v21_COSINE.ipynb`; xem `NOTEBOOK_GUIDE_COSINE.md`.** Hướng dẫn dưới
đây mô tả recipe hyperbolic đã chạy ở ZIP5, được giữ để tái lập.

File notebook giữ tên `run_vadclip_v21_PAPER.ipynb` để dùng cùng
vị trí versions, nhưng recipe hiện tại là **SANITY_FIXED_C**, không phải mã tác giả
PoinCLIP-VAD đã xác minh1:1. Run ZIP5 `run_4jua44k6` đã được kiểm: UCF C-AUC
80.0597%, XD A-AP49.6146%, đều thấp hơn V0; đọc phân tích trong
`../kaggle_result/V21/ANALYSIS_run_4jua44k6.md`. Không resume
checkpoint schema v4 của các run cũ; schema hiện hành là v5_sanity.

## Luồng mô hình và thay đổi

Ảnh: CLIP feature.npy → local Transformer → global GCN → FC → conditioning riêng
visual → scale học được → smooth radial compression → expmap0.
Text: prompt học được + class token → frozen CLIP text → conditioning riêng text →
scale học được → smooth radial compression → expmap0. Gradient vẫn về prompt.
Classification dùng sigmoid(-distance tới prototype+bias). Visual prompt tổng hợp
ở tangent space, normalize, FFN+text residual rồi expmap0. Alignment dùng -distance/.07.
Không chạy lại CLIP image encoder, không dùng cosine head thay Poincare.

- Temperature áp đúng một lần trong alignment map; CE không chia lại; evaluator
  nhận cùng đơn vị logits. MIL pooling ở train vẫn khác scoring từng frame theo thiết kế.
- Conditioning lấy median của các median norm mỗi video train, và median norm các
  class text; chọn tối đa32video train, normal/abnormal và có phủ lớp khả dụng.
  Lưu hai norm vào checkpoint buffers; không L2-normalize từng vector, không đổi
  FC weights để calibrate, không lấy thống kê test. Lambda ban đầu vẫn .7 trên
  feature đã conditioning. Bias classifier lấy median train distance sau conditioning.
- Cố định c=.5 toàn run, không có gradient/optimizer state cho c.
- Nén norm tangent trơn bằng tanh, cap relative radius=.95; khởi tạo điển hình gần
  .46 nếu norm sau conditioning gần1. Đây là stabilization có công bố, không phải
  công thức paper nguyên văn. Outlier cực lớn vẫn có thể mất radial gradient;
  cần đọc health log, không coi bounded radius là bảo đảm học tốt.
- Cone exterior angle dùng atan2(perpendicular norm, parallel), bỏ conservative
  acos clamp từng chặn gradient ở probe FP64. Giữ guards aperture/degenerate và
  log legacy_acos_would_clip_fraction: đây là tỷ lệ clamp cũ SẼ kích hoạt, không
  phải clamp atan2 đang sử dụng. Hinge inactive vẫn có thể có regularizer gradient.
- Adam LR5e-4, WD5e-5 cho matrices, zero WD cho scalar/bias/gain vectors; giữ betas
  (.9,.999) để chưa thêm một ablation optimizer nữa. Clip finite global gradnorm1.
- Encoder/fusion và BCE/CE FP32, manifold/cone/distance loss FP64 (tổng loss FP64);
  AMP tắt. Không cam kết VRAM hoặc thời gian
  từ unit tests. Mốc30epoch/cosine giữ nguyên; chưa thêm LR warmup hoặc delayed entailment.
- Giữ psi=.001, phiUCF=.1/XD=.0001; không tự tăng entailment weight theo MERU.

## Input Kaggle

Gắn6gói như run trước:
1. UCF-Crime test290, feature ViT-B/16 và metadata theo package V0 đã khóa.
2. Checkpoint gốc UCF `model_ucf.pth`.
3. VadCLIP XD-Violence Test800, đủ feature/metadata package V0 đã khóa.
4. Checkpoint gốc XD `model_xd.pth`.
5. `vadclip_ucf_train_v5.zip`.
6. `vadclip_xd_train_v5.zip`.

Bật GPU và Internet. `ViT-B-16.pt` có thể gắn bổ sung để tránh tải CLIP; code kiểm hash.
Các override đường dẫn để None sẽ autodetect; khi có nhiều input trùng tên hãy điền
UCF_DATA_ROOT/UCF_CHECKPOINT/XD_DATA_ROOT/XD_CHECKPOINT/UCF_TRAIN_ROOT/XD_TRAIN_ROOT
đúng file/root của Kaggle. Không thay manifest hashes để hợp thức hóa input khác.

## Ba cách chạy

- `RUN_MODE="smoke"` mặc định: kiểm thực thi, physical batch64, tối đa2batch/1epoch,
  inference vài video. Không đọc metric smoke như benchmark.
- `RUN_MODE="pilot"`: train3epochs trên đầy đủ official train của cả hai dataset;
  KHÔNG gọi evaluator official test, KHÔNG sinh AUC/AP/latency test. Đọc learning_health
  và checkpoint/log. Không dùng pilot để tuyên bố phục hồi localization.
- `RUN_MODE="full"`: V0 evaluation với anchor đã khóa → train mới30epochs → V21
  evaluation → paired comparisons hai dataset. Final epoch cố định; không chọn trên test.

Mỗi lần chạy tạo run folder riêng, không ghi đè run cũ. Pilot là diagnostic riêng;
đổi pilot sang full sẽ train mới từ đầu, không auto resume. Full không có dev selection;
nếu cần chọn hyperparameter, phải thiết kế train/development trước, không dùng bảng test.

## Đọc output trước full tiếp theo

`ucf/train/v21/` và `xd/train/v21/` có:
- `preflight.json`: video train đã dùng, conditioning norms, calibrated bias, health probes.
- `encoder_flow.json`: trace module/source/hash và gradient thực của local/global/text.
- `learning_health.json`: hai effective batch đầu mỗi epoch, per-loss raw/weighted
  gradient norm về adapter/text (mean loss mỗi microbatch), radius min/median/p95/max,
  norm tangent trước compression/soft limit, cone guard/hinge stats,
  std raw C logits/C probabilities/alignment log-odds trên snippet hợp lệ từng video.
- `history.json`: loss từng thành phần, updates, c/scales, LR; fixed c phải giữ nguyên.
- `gradient_diagnostics.json`: finite-gradient check và global norm/update.
- `best.pth`: downstream+norm buffers, final epoch theo mode. Không kèm frozen CLIP.

Không đặt luật variance luôn phải lớn hoặc hinge luôn phải active. Hai số đó không
thay thế kiểm gradient trực tiếp, đặc biệt với video normal hoặc hinge đã thỏa.
Finite checks/radius contract sai sẽ fail rõ; các dấu hiệu học kém được log để review,
không tạo metric giả hoặc bỏ batch. Không có bảo đảm benchmark từ smoke/pilot.

Cell cuối cung cấp compact `vadclip_v21_results.zip`, kể cả partial run nếu lỗi và disk
cho phép. Để gửi kết quả, dùng file ZIP này hoặc bản sao `v21_run_*_results.zip` từ cell
cuối. Kaggle Download All còn có training-state/cache và có thể lớn hơn nhiều.

## Những giới hạn vẫn còn

Không có code tác giả PoinCLIP-VAD để chứng nhận init/pair mining/normalization,
window layout và feature grid1:1. Scalar conditioning, fixedc, smooth limiter, atan2,
bias init và optimizer groups là lựa chọn tái hiện/stabilization công bố rõ.
Luồng inherited VadCLIP vẫn bỏ padding_mask trong temporal call, dùng lengths ở
feature adjacency; bản này giữ hành vi upstream. Mask ở loss/prompt/diagnostics
không chứng minh temporal toàn mạng padding-invariant. Cần ablation riêng nếu sửa.
Các official test đã được xem nhiều lần; công bố test exposure khi báo kết quả.

Nguồn tham khảo:
- PoinCLIP-VAD: https://www.nature.com/articles/s41598-026-56792-z
- MERU mục4.1: https://proceedings.mlr.press/v202/desai23a/desai23a.pdf
- HyCoCLIP AppendixB: https://proceedings.iclr.cc/paper_files/paper/2025/file/d9dd691fa4eeb4741f16ac42ed5d197e-Paper-Conference.pdf
