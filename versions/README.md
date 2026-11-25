# Chạy trên Kaggle

| Notebook | Cấu hình trong lượt full | Feature train |
|---|---|---|
| [V00.ipynb](V00.ipynb) | VadCLIP V0 | — |
| [V07.ipynb](V07.ipynb) | V0 và V7 | — |
| [V20.ipynb](V20.ipynb) | V0, V7, V7 + temporal và V20 | UCF Train1610 V5, XD Train3954 V5 |

Mỗi notebook nhúng runtime và được import trực tiếp lên Kaggle. Gắn UCF-Crime
Test290 (crop `__5.npy`), XD-Violence Test 800 (crop `__0.npy`) và hai checkpoint
`model_ucf.pth`, `model_xd.pth`. V20 cần thêm hai gói train trong bảng và tự
huấn luyện head trên dữ liệu train. Bật GPU và Internet, đọc cell config, rồi
chạy smoke trước khi chấm full. Cell cuối dùng để tải ZIP.

[Metric và nguồn của các lượt full](../results/README.md).
