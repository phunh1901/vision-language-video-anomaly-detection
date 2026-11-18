# V16 — Neo Normal thích ứng theo video (lộ trình mới, 14/11/2026)

**Đây là V16 mới.** Thử bước 1 trước: đóng băng toàn bộ checkpoint V0, chỉ thay
vector Normal cuối của A-Branch bằng hỗn hợp với feature bình thường trong cùng video.
Không huấn luyện, không thay prompt lớp bất thường, không thay LGT, không trộn hai nhánh.
Neo Normal gốc của VadCLIP đã có visual prompting theo block; V16 giữ neo gốc đó
làm fallback và thêm một neo nhìn lấy từ toàn video. Không coi neo gốc là text cố định thuần túy.

## Bốn input bắt buộc — đúng các gói đã chạy V0/V7

| Input | Nội dung |
|---|---|
| UCF test | `ucfclipfeatures_test290.zip` (290 video, crop 5) hoặc cây feature đã giải nén |
| Checkpoint UCF | `vadclip_ucf_checkpoint_upload.zip` chứa `model_ucf.pth`, hoặc file `.pth` |
| **VadCLIP XD-Violence Test 800** | `vadclip_xd_test800_kaggle_v2.zip` và manifest đã dùng ở V0; tên Dataset giữ nguyên |
| Checkpoint XD | `vadclip_xd_checkpoint.zip` chứa checkpoint V0, hoặc file `.pth` |

**Không cần output V15, train features, I3D, video MP4 hoặc audio.** Dùng Add Input
gắn các Dataset cũ; không cần đổi tên Dataset. Nếu gắn nhiều bản cùng loại, điền
đường dẫn cụ thể ở cell cấu hình. Không tắt kiểm hash để vượt lỗi input.

Kaggle: GPU T4, Internet ON để lấy mã VadCLIP commit đã khóa và CLIP ViT-B/16.
`CLIP_WEIGHTS` tùy chọn là đường dẫn `ViT-B-16.pt` OpenAI chính thức; không phải checkpoint VadCLIP.

## Cách chạy

1. Import notebook; để `RUN_MODE = "smoke"`; Run All.
2. Kiểm cả `ucf` và `xd` báo `ok`. Smoke đi qua input/hash, checkpoint thật và tính neo.
3. Đổi **duy nhất** `RUN_MODE = "full"`; Run All. Giữ cấu hình trước khi xem kết quả.
4. Tải `/kaggle/working/vadclip_v16/run_*/vadclip_v16_results.zip`.

Mỗi bộ chạy V0, đối chứng `identity` với beta=0, rồi V16. Full buộc identity khớp
bitwise cả hai nhánh và C-V16 khớp C-V0; dừng nếu V0 không khớp mốc tái lập.

## Công thức đã khóa cho phép thử đầu

Chỉ chọn snippet có sigmoid(C) ≤ 0,20 và 1-softmax(A)[Normal] ≤ 0,20,
đồng thời các hàng xóm thực liền kề cũng đạt. Lấy tối đa 20% snippet điểm thấp nhất,
cần ít nhất 8 snippet và cosine coherence ≥ 0,50. Tất cả lựa chọn chỉ dùng prediction,
không nhận nhãn video/frame test. Các ngưỡng là giả định thăm dò, chưa hiệu chuẩn.

Gom feature thị giác **sau adapter, trong đúng không gian dùng để tính A**, trọng số
(1-pC)(1-pA), rồi chuẩn hóa. Không thêm projection chưa được học. Beta tối đa 0,10,
giảm theo độ tin cậy, lượng bằng chứng và coherence. Không đủ điều kiện thì giữ nguyên.
Chỉ thay logit Normal cuối, giữ temperature gốc và công thức 1-softmax(A)[Normal].
Toàn bộ inference dùng no_grad/detach. Các block cùng một video chia sẻ neo bổ sung;
không trộn neo giữa video; loại padding trước khi chọn và gom feature.

Đây là phép suy luận **offline** dùng toàn video, không phải phát hiện nhân quả trực tuyến.

## Đọc kết quả

- `comparison.csv`: A-V16 so A-V0 và C-V16 so C-V0, AUC/AP hai bộ, Ano-AUC UCF.
- UCF lấy C làm mốc chính nên V16 này **phải giữ nguyên AUC chính UCF**. Việc A tăng
  chưa chứng minh hệ thống vượt mốc chính; không tự đổi sang nhánh tốt hơn sau khi xem test.
- `anchor_diagnostics.json`: beta, số snippet, chỉ số được chọn và lý do fallback từng video.
- `predictions.npz`: điểm hai nhánh, ID, offset và GT để replay; `run.json` lưu hash/metric.
- `false_alarm_diagnostics.csv`: FPR tại recall yêu cầu 50%/80%, kèm recall thực do ties.
  Ngưỡng này dùng nhãn test để phân tích hồi cứu, **không dùng triển khai hoặc chọn cấu hình**.
- CI bootstrap 500 lần lấy mẫu theo video, không coi các frame độc lập; không phản ánh nhiều seed train.
- Không có temporal mAP được kiểm chứng trong phép thử này; không tuyên bố bảo toàn fine-grained.

## Thời gian và giới hạn

Không có lượt train. Mỗi bộ chạy ba lượt inference và kiểm hash input.
Smoke in ước lượng full từ thời gian đo thật; độ dài video khác nhau làm ước lượng có sai số.
Dự trù ban đầu 10–30 phút trên T4 với input sẵn; đây chưa phải thời gian V16 đo trên Kaggle,
không gồm tải CLIP/mạng chậm. Nếu smoke lỗi, gửi ZIP/log trước khi chạy full.

## Các bước kế tiếp

V17: điều biến residual C→A, đối chứng riêng và kết hợp với V16.
V18: cổng biến thiên ngữ nghĩa trực chuẩn, chỉ điều biến cập nhật thời gian.
V19: hard-negative prompts với loss margin và đối chứng giữ nguyên công thức điểm.
Ba bước này **chưa được triển khai/huấn luyện trong notebook V16**; không ghép trước khi đọc kết quả bước 1.
