# V21 — notebook chính duy nhất

Chạy `V21.ipynb`. Các notebook V21 cũ đã chuyển vào archive; không cần tải các file .py lên Kaggle vì notebook nhúng sẵn mã nguồn.

Mục tiêu hiện tại: dựng lại cosine theo **code mặc định ở commit c41067f**, trước khi đưa Hyperbolic trở lại. Đây là thay đổi toàn bộ recipe so với ZIP7, không phải đối chứng chỉ đổi optimizer. Không khẳng định tái lập chính xác lịch sử train của checkpoint phát hành hoặc bảo đảm đạt V0.

## Input và cách chạy

Giữ input test UCF, “VadCLIP XD-Violence Test 800”, hai checkpoint V0 và CLIP như trước. Không cần output ZIP cũ.

**Thay hai gói train v5 bằng feature train gốc đủ 10 crop/video.** UCF phải có 16.100 file, XD 39.540 file theo tên gốc `...__0.npy` đến `...__9.npy`. Hỗ trợ một ZIP hoặc thư mục đã giải nén cho mỗi dataset. Không hỗ trợ ZIP lồng trong ZIP hoặc chia nhiều part. Hai gói `vadclip_*_train_v5.zip` hiện tại chỉ có một crop/video, không đủ.

Mặc định UCF_TRAIN_ROOT/XD_TRAIN_ROOT=None: notebook tự tìm feature gốc đủ crop trong /kaggle/input (ZIP hoặc thư mục giải nén), không cần tự điền đường dẫn. Nếu nhiều bản trùng nhau hoặc muốn chỉ định rõ nguồn, điền hai đường dẫn Kaggle:

```python
UCF_TRAIN_ROOT = "/kaggle/input/TEN-INPUT-UCF/UCFClipFeatures"  # hoặc ZIP gốc
XD_TRAIN_ROOT = "/kaggle/input/TEN-INPUT-XD/XDTrainClipFeatures"  # hoặc ZIP gốc
RUN_MODE = "full"
```

Notebook lưu input_discovery.json với số feature/crop tìm thấy, phân biệt gói v5 một crop với gói gốc. Nếu chỉ có gói v5, báo rõ số video và crop index còn thiếu, không báo lỗi chỉ vì biến đường dẫn=None.

Notebook kiểm tra **cả hai tập trên CPU trước bất kỳ job GPU nào**: tên/đủ crop, nhãn CSV, shape Nx512, hữu hạn và SHA-256 từng file; ghi crop_preflight/preflight.json và validated_crop_inventory.json. Thiếu crop thì dừng kèm ví dụ file thiếu, không tự lặp một crop để giả đủ dữ liệu.

`smoke`: kiểm tra toàn bộ input rồi chạy hai update/tập và benchmark smoke. `pilot`: train-only ba epoch với cùng mốc MultiStepLR, không benchmark official test. Pilot khởi tạo lại, không tự nối sang full. `full`: 10 epoch/tập, checkpoint cố định cuối, so V0 trên cả hai tập.

## Recipe được chọn: mặc định repo, không phải bảng hyperparameter trong paper

| Thiết lập | UCF | XD |
|---|---:|---:|
| Số crop/video | 10 | 10 |
| Effective batch | 128 (64 normal + 64 abnormal) | 96 (shuffle toàn tập) |
| Updates/epoch | 125 | 412 |
| Epoch full | 10 | 10 |
| Tổng updates full | 1.250 | 4.120 |
| LR | 2e-5 | 1e-5 |
| MultiStepLR, gamma 0.1 | [4, 8] | [3, 6, 10] |

AdamW weight decay 0.01 theo default code gốc; không clipping gradient, FP32, seed 234. Video dài được lấy trung bình từng bin thời gian theo process_feat mặc định. Giữ native forward, native MIL/CLASM và hệ số loss theo nguồn đã khóa; không đổi BCE sang objective mới. Không tự hạ batch khi OOM vì sẽ thay recipe.

Khác biệt còn công khai: thứ tự shuffle bằng NumPy thay torch DataLoader; probes train-only và reset RNG; FP32 encoder; chọn final epoch thay chọn checkpoint bằng official test. Vì vậy không dùng nhãn “exact reproduction”.

## Kết quả và chẩn đoán

- `crop_preflight/`: toàn bộ inventory, số crop/video, batch, updates dự kiến và source defaults.
- `train/`: split_manifest, initialization, native_objective, history, training_flow, optimizer_flow, learning_health và độ phủ crop/temporal bin.
- `error_analysis/per_video_errors.csv`, `error_segments.csv`, `stage_evidence.csv`: mẫu sai, đoạn FP/FN và bước quan sát lỗi, phân biệt bằng chứng đầu ra với nguyên nhân chưa chứng minh.
- `ranking_tail_summary.csv`, `score_distributions.csv`, `top_false_positive_segments.csv`, `top1pct_false_positive_videos.csv`: đuôi top 1%, xử lý tie theo trọng số, các đoạn nền điểm cao.
- `characteristic_summary.csv`, `paired_video_errors.csv`: nhóm đặc điểm và mức thay đổi so V0.

Lưu epoch 3/6 riêng trên Kaggle, final ở best.pth (tên tương thích, không có test-based best selection). ZIP mặc định chỉ gồm final weights/predictions và các báo cáo. Bật INCLUDE_EPOCH_CHECKPOINTS nếu cần đưa cả epoch 3/6 vào ZIP. Không suy ra overfit từ train loss khi chưa có đánh giá checkpoint sớm.

Chi phí dự kiến: tổng 5.370 updates so với 2.730 của ZIP7, batch lớn hơn và nhiều I/O hơn. Chưa có đo thời gian full mới; không dùng 42 phút của ZIP7 làm cam kết runtime. Có thể OOM nếu GPU không đủ batch native; lỗi được ghi lại, không âm thầm đổi batch.
