# V21 — phương án B: train downstream mới

**Trạng thái 20/11/2026: B đã chạy GPU và không đạt mục tiêu. Dừng chạy full lặp lại.**
Run__si5up5p: UCF AUC81.3917%, XD AP62.4797%; predictions replay khớp báo cáo.
Audit phát hiện XD train chỉ có1video mang nhãn Abuse, dev không có Shooting/Abuse.
Source hiện đã bổ sung split_health.json và chặn train khi train/dev thiếu lớp.
Cần giải quyết giao thức/đối chứng trước khi train tiếp; không tắt guard để chạy.


Notebook độc lập; không cần upload helper/source riêng. Đây là biến thể triển khai
theo PoinCLIP-VAD, chưa phải tái lập code tác giả. Không cam kết tăng điểm.

## Khởi tạo được kiểm tra bằng mã

- Khởi tạo mới temporal adapter/GCN, prompt, fusion và head; không nạp downstream V0
  hoặc checkpoint V21 trước. CLIP pretrained đóng băng, feature ảnh .npy có sẵn.
- V0 vẫn cần để đối chiếu baseline và nạp đúng các tensor frozen `clipmodel.*`.
  Khi nạp CLIP, hash downstream trước/sau phải giống nhau; lưu initialization.json.
- Không gọi reset_parameters đệ quy lên CLIP. Không dùng no_grad cho text encoder
  khi học prompt đầu vào; weights CLIP không nhận gradient.

## Lịch 30 epochs

| Epoch | Thành phần học | LR |
|---|---|---|
| 1–3 | Toàn downstream, trừ curvature và scale đang cố định | Tăng tuyến tính tới1e-4 |
| 4–30 | Toàn downstream, mở curvature/scale | Cosine1e-4→1e-6; c/scale1e-5→1e-7 |

Pha đầu vẫn dùng khoảng cách Poincaré ở curvature nhỏ, không gọi là cosine loss.
c init0.05, khoảng(1e-5,2), lambda visual/text0.7, temperature0.07. Giữ lambda0.7
để không ép toàn bộ embedding về sát gốc khi còn dùng entailment cone; không dùng
L2normalize toàn feature. Median norm projector hiệu chỉnh trên tối đa8video train.
Bias classifier mới được khởi tạo bằng median distance trên cùng các video train,
để median logit ban đầu bằng0 thay vì toàn bộ lệch âm; không học từ dev/test.
Đây là các lựa chọn triển khai công khai, không tự nhận là hyperparameters tác giả.

Giữ classifier prototype-distance, fusion, loss và split/exclusion của A để tránh
đổi thêm kiến trúc. Vì B đổi cả initialization/lịch và hiệu chỉnh bias/c, không được
gán mọi chênh lệch điểm cho riêng checkpoint warm-start. XD vẫn giữ source exclusion
đã khóa; không gọi là học trên toàn3954video như paper. Geometry/loss FP32, AMPFalse,
gradient clipping1, tangent clipping norm3 cả sau FFN. Scalars/bias không decay.

## Kaggle và input

GPU T4/Internet; gắn sáu input như A:
1. UCF test290 feature ZIP.
2. model_ucf.pth gốc (hoặc ZIP checkpoint).
3. VadCLIP XD-Violence Test 800.
4. model_xd.pth gốc.
5. vadclip_ucf_train_v5.zip.
6. vadclip_xd_train_v5.zip.

Không cần output V21 cũ. CLIP ViT-B-16.pt đúng hash có thể gắn thêm để tránh tải.
Chạy RUN_MODE="smoke" trước: hai epoch, mỗi pha1, tối đa2macro-batches/epoch.
Smoke phải kiểm prompt/adapter có gradient ngay pha1, c/scale không thay đổi ở pha1,
rồi mở ở pha2. Sau đó đổi RUN_MODE="full", chạy lại từ đầu, không nối checkpoint smoke.

## Kết quả, lỗi và giới hạn

V0 inference → train B → evaluate checkpoint best development; cả UCF/XD, C/A riêng.
Chọn UCF theo dev C-video AP, XD theo dev A-video AP; đây là proxy khác frame metric.
Không dùng official test để chọn checkpoint. Log recipe B, pha, LR theo nhóm và init hash.
ZIP giữ score predictions để chẩn đoán, checkpoint downstream, config/source/split/history,
metrics/latency/logs. Không giữ feature gốc/CLIP trong ZIP. Có cell tải output cuối.
Lỗi runtime dừng và giữ ZIP partial khi có thể; không báo lỗi thành thành công.
Hết disk/kill session không thể bảo đảm lưu; snapshot cuối chưa hỗ trợ auto-resume.
Latency chỉ feature-trên-GPU→logits; không phải E2E/raw-video FPS. Thời gian/VRAM phải đo
trên Kaggle; lượt B vừa chạy mất khoảng160phút toàn pipeline. Không hứa30epoch trong30phút.
