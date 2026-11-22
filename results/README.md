# Kết quả và kinh nghiệm

Mỗi version có thư mục run. Git chứa notes.md và metrics.json đã rà soát; ZIP gốc giữ local.

| Version | Gói ZIP gốc | Nơi đọc log/kết quả/kinh nghiệm |
|---|---|---|
| V00 | 1 | [run_gs1ntqbt](V00/run_gs1ntqbt/notes.md) |
| V01 | 0 | [historical_local](V01/historical_local/notes.md) |
| V02 | 0 | [historical_local](V02/historical_local/notes.md) |
| V03 | 0 | [historical_local](V03/historical_local/notes.md) |
| V04 | 0 | Chưa tìm thấy artifact; không suy là chưa chạy |
| V05 | 0 | Chưa tìm thấy artifact; không suy là chưa chạy |
| V06 | 0 | Chưa tìm thấy artifact; không suy là chưa chạy |
| V07 | 1 | [historical_local](V07/historical_local/notes.md); [run_tutkhmvv](V07/run_tutkhmvv/notes.md) |
| V08 | 0 | Chưa tìm thấy artifact; không suy là chưa chạy |
| V09 | 0 | [historical_local](V09/historical_local/notes.md) |
| V10 | 0 | [historical_local](V10/historical_local/notes.md) |
| V11 | 1 | [run_cbnev29b](V11/run_cbnev29b/notes.md) |
| V12 | 1 | [run_t7ve8ptu](V12/run_t7ve8ptu/notes.md) |
| V13 | 1 | [run_130tnvql](V13/run_130tnvql/notes.md) |
| V14 | 1 | [run_w_je5uq0](V14/run_w_je5uq0/notes.md) |
| V15 | 1 | [run_au3gpix8](V15/run_au3gpix8/notes.md) |
| V16 | 1 | [run_eljfkmkf](V16/run_eljfkmkf/notes.md) |
| V17 | 1 | [run_vj38i365](V17/run_vj38i365/notes.md) |
| V18 | 1 | [run_55zyk2nz](V18/run_55zyk2nz/notes.md) |
| V19 | 1 | [run_tran7l08](V19/run_tran7l08/notes.md) |
| V20 | 1 | [run_259y0a3x](V20/run_259y0a3x/notes.md) |
| V21 | 9 | [historical_local](V21/historical_local/notes.md); [run_4__dwo_a](V21/run_4__dwo_a/notes.md); [run_4jua44k6](V21/run_4jua44k6/notes.md); [run__si5up5p](V21/run__si5up5p/notes.md); [run__zzcyv2i](V21/run__zzcyv2i/notes.md); [run_bg0maqz8](V21/run_bg0maqz8/notes.md); [run_c5ieenya](V21/run_c5ieenya/notes.md); [run_djs32hmj](V21/run_djs32hmj/notes.md); [run_yxfrilug](V21/run_yxfrilug/notes.md) |

- result.zip: ZIP tải nguyên gốc, giữ SHA-256 và tên gốc trong notes.
- analysis.zip: gói phân tích đi kèm nếu có.
- historical_local/extra.zip: đóng gói local phần độc nhất đã giải nén; có manifest/hash, không giả là run hoặc ZIP tải gốc.
- Nếu chưa tìm được run_id, dùng unassigned_<sha> và giữ trạng thái chưa xác định.
- Chỉ giải nén khi cần; giữ dữ liệu độc nhất và kết quả âm. Không lấy source hiện tại thay snapshot của run cũ.
- Thời gian/metric đọc JSON chưa phải replay độc lập; không suy ngày chạy từ mtime.

metrics.json lưu nguồn member và SHA-256 archive; file nhị phân không đi kèm bản clone Git.
