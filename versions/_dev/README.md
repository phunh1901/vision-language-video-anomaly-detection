# Mã phát triển

Mở notebook V00–V21 ở thư mục cha để chạy Kaggle; không cần upload riêng source Python.
V0–V15 có runtime nhúng trong notebook. v16–v21 giữ source và builder còn dùng;
common/clip_vad chứa duy nhất helper latency/reporting; templates là template lịch sử V14 mà V18 cần.

Builder V21: `python versions/_dev/v21/build_notebook.py COSINE`.
Builder này sinh versions/V21.ipynb. Recipe A/B/PAPER là fixture lịch sử, không ngang hàng
với notebook chạy chuẩn. Kiểm diff và payload khi sinh lại; không tự thay code của run cũ.

Cấu hình nằm ở cell notebook và notebook_config*.py cạnh builder. Không có thêm lớp config registry.
Test ở tests/, hashes notebook chuẩn ở tests/fixtures/notebook_hashes.json.
