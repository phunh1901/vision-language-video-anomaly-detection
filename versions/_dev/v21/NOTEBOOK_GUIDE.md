# VadCLIP V21 — phương án A: fine-tune hai giai đoạn

## Phương án A — 19/11/2026 (cấu hình hiện hành)

- Full: 16 epochs = 4 head warm-up + 12 joint fine-tune. Smoke: 2 epochs,
  mỗi pha 1 epoch và tối đa 2 macro-batches/epoch, để kiểm tra mở khóa.
- Pha 1: adapter và prompt/fusion V0 đóng băng, đặt eval để tắt dropout/state updates.
  Chỉ projector, classifier, curvature và scale học. Head LR tăng tuyến tính theo epoch
  tới 1e-4; curvature/scale tới 1e-5. Giữ CLIP đóng băng cả hai pha.
- Pha 2: mở toàn bộ downstream; LR adapter/prompt=2e-5, head=5e-5,
  curvature/scale=5e-6. Cosine từ giá trị này tới 2% ở epoch cuối (từng nhóm).
  Giữ nguyên optimizer và Adam moments khi chuyển pha; không reset mô hình.
- Curvature vẫn init=0.5, trong (0.1,1); scale ảnh/chữ init=0.7; tau=0.07.
  Không áp curriculum c=0.01 của phương án B. Geometry/loss FP32, AMP=False.
- Chặn norm tangent tối đa 3 trước expmap ở cả video, text, prototype và sau FFN.
  Không normalize mọi mẫu thành cùng bán kính. Đây là biện pháp triển khai để ổn định,
  không phải thông số tác giả đã xác minh; log max_tangent_norm trong geometry receipt.
- Giữ BCE miền log đã sửa. Preflight vẫn chỉ dùng train, không có optimizer step.
  Kiểm gradient theo pha; không yêu cầu gradient adapter/prompt khi chúng bị đóng băng.
- Best checkpoint chọn trên development qua cả hai pha; chứa train_config/training_phase.
  Snapshot cuối lưu optimizer/scaler/RNG/lịch pha; chưa hỗ trợ auto-resume.
- Lỗi runtime được ghi log/failure.json nếu trainer còn hoạt động; dừng dataset lỗi,
  lưu ZIP phần đã có. Cell tải lấy được ZIP partial sau lỗi. Không nuốt lỗi để báo đạt.
  Kill session, mất điện hoặc hết disk không bảo đảm ghi được artifact.
- Đây là adaptation theo PoinCLIP-VAD với lịch A được đề xuất; không tuyên bố tái lập
  chính xác mã tác giả, không cam kết tăng benchmark. Chưa chạy GPU bản A.

## Nền sửa sau lượt lỗi run_bg0maqz8 — 19/11/2026

- BCE của mean(top-k sigmoid) được tính tương đương bằng logsumexp/logsigmoid,
  bỏ clamp xác suất gây mất gradient. Không thay thành BCE của mean logits.
- Mặc định AMP=False: downstream/loss float32; CLIP frozen giữ dtype nội bộ.
- Trước train có preflight tối đa 8 video train (4 normal/4 abnormal), không cập nhật
  optimizer, kiểm forward/backward hữu hạn và gradient BCE classifier khác 0.
- Khởi tạo hai projector được hiệu chỉnh bằng median norm trên chính mẫu train này:
  median norm sau projector đặt về 1, trước lambda=0.7. Đây là lựa chọn ổn định số
  của V21, không phải thông số tác giả đã công bố; không dùng test/dev để hiệu chỉnh.
  Lưu norm trước hiệu chỉnh và bảng BCE/logit/radius tại preflight.json và cell output.
- Kiểm tensor không hữu hạn trong training trước BCE; không thêm các kiểm tra đồng bộ
  này vào forward inference để tránh làm sai phép so latency.
- STOP_ON_FAILURE=True: một dataset lỗi thì đóng gói kết quả và dừng, không tự chạy
  thêm dataset còn lại. Không che lỗi bằng bỏ video hoặc trả metric giả.
- Schema checkpoint đổi sang v2; train mới từ V0, không nạp best/optimizer của lượt lỗi.
- Giữ 30 epoch/Adam/lr/WD/batch64, distance/tau một lần và protocol test hiện có.
  Giữ lọc giao nhóm nguồn: XD vẫn chỉ còn 1.600 video đủ điều kiện; không gọi đây là
  tái lập benchmark paper trên toàn bộ official train.
- Chạy RUN_MODE="smoke" trước; khi đạt đổi "full". Full cũng tự chạy preflight.
  Preflight không bảo đảm mọi batch tương lai hoặc benchmark sẽ đạt; lỗi phát sinh
  vẫn được báo rõ. Bản sửa chưa có kết quả GPU/benchmark mới.

Trên mỗi dataset: **V0 inference → train V21 Poincaré → V21 inference**.
Không train cosine control theo yêu cầu ngày 19/11/2026. Benchmark paper là động lực;
V21 chỉ có kết luận sau chạy thật. Không tự gán mức tăng của paper cho bản triển khai này.

## Input Kaggle — sáu gói như V20

1. UCF test290 features (`ucfclipfeatures_test290.zip`).
2. Checkpoint gốc `model_ucf.pth` hoặc ZIP checkpoint UCF.
3. `VadCLIP XD-Violence Test 800` (feature ZIP chuẩn bị trước).
4. Checkpoint gốc `model_xd.pth`.
5. `vadclip_ucf_train_v5.zip`.
6. `vadclip_xd_train_v5.zip`.

Không cần output V7/V20. CLIP ViT-B-16.pt gốc có thể gắn thêm để tránh tải.
Bật Internet và GPU Tesla T4. Tự dò input hoặc điền đường dẫn tại config.

## Những phần được học và hình học

- CLIP pretrained luôn đóng băng. Pha1 chỉ học geometry/classifier; pha2 học thêm
  adapter temporal/GCN, prompt parameters và FFN fusion.
- Projector d→d rồi expmap vào Poincaré ball, curvature=-c; không thêm chiều Lorentz.
  Nhánh C: sigmoid(-Poincaré distance tới prototype + bias).
- Prompt thị giác dùng attention C và logmap; FFN fusion trong tangent space rồi expmap.
  Nhánh A: -Poincaré distance/tau, nhiệt độ chỉ áp dụng một lần.
- Loss: BCE classification + video-label alignment + psi*entailment + phi*separation.
- Full: **16 epochs (4 + 12), Adam với LR theo nhóm/pha ở trên, WD=5e-5, cosine pha2, batch hiệu dụng 64**
  (32 normal/32 abnormal). Microbatch 1, tích lũy gradient để giảm VRAM.
  Mặc định downstream float32; CLIP giữ dtype nội bộ, loss float32. Đây là batch hiệu dụng qua
  accumulation, không khẳng định tốc độ/tương đương mọi phép toán với batch vật lý 64.
- Smoke: 2 epochs (1 mỗi pha), tối đa 2 macro-batches/epoch, batch hiệu dụng 4, dev nhỏ; không chấm benchmark.
- Train sample tối đa 256 snippets/video. Dev/test dùng đủ snippet qua block gốc.

## Checkpoint và giới hạn

Chia train/dev theo nhóm nguồn, loại nhóm tên giao test, chỉ dùng video labels để học.
Chọn epoch bằng dev **C-video AP trên UCF**, **A-video AP trên XD**, chạy đủ 16 epoch;
không dùng frame test labels để chọn. Warm-start adapter/prompt từ V0, thay C classifier.

Đây là adaptation downstream đầy đủ theo paper, **chưa phải tái lập chính xác code
tác giả** hoặc train từ đầu. Chưa tìm được repository tác giả trong nguồn đã kiểm.
Paper chưa đủ chi tiết về cone aperture, entailment regularizer, margin/pair mining;
giá trị config là lựa chọn triển khai công khai. Init projector identity/sqrt(d), c=0.5
và gradient clipping cũng là lựa chọn nội bộ. Giữ feature/split/evaluator VadCLIP hiện có.
Trunk V0 từng học train gốc gồm cả dev vừa tách; official test đã xem ở các version trước.
Kết quả là exploratory. Không có cosine retraining nên so V0/V21 đo toàn bộ thay đổi,
chưa tách riêng đóng góp của khoảng cách hình học khỏi tác động huấn luyện/kiến trúc.

## Log, latency và output

Cell in bảng căn cột cấu hình, từng epoch, hai nhánh C/A, primary metric, latency,
VRAM/params và delta so V0 cùng lượt. Bốn thành phần loss lưu trong history.json.
Bootstrap theo video cho cả C/A. Không ép C bất biến như bản cũ. Hash checkpoint
khởi tạo V0 và checkpoint thực đánh giá được ghi riêng.

Latency đo feature đã trên GPU → logits với CUDA sync/warmup/repeats. Không gồm decode
hay CLIP image feature extraction; không gọi end-to-end FPS. Giữ boundary như V0/V7/V20.
Không tính FLOPs, Base/Novel hay online delay.

ZIP chứa config/source/hash/split/history/metrics/latency/log và `best.pth` downstream.
Không chứa CLIP weights; cần V0 gốc để nạp lại. `last_training_state.pt` giữ trong thư
mục Kaggle, không đưa ZIP; snapshot phục hồi thủ công, chưa có auto-resume.
Predictions chỉ đưa ZIP khi INCLUDE_PREDICTIONS=True. Cell tải dùng link tương đối.

Lý thuyết: https://doi.org/10.1038/s41598-026-56792-z
