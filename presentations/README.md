# Slide báo cáo benchmark

- [PowerPoint](quantum_runtime_benchmark_vi.pptx): 13 slide, gồm 10 slide chính
  và 3 slide phụ lục. Có speaker notes dẫn tới nguồn số liệu.
- [PDF](quantum_runtime_benchmark_vi.pdf): bản xuất để xem nhanh.
- [Biểu đồ và nguồn](figures/README.md): bốn biểu đồ, mỗi biểu đồ có PNG và SVG.
- [Nội dung khoa học](../docs/presentation.md): phạm vi, kết quả và giả định.
- [Bảng số liệu](../artifacts/benchmark_v1/results/benchmark_summary/two_domain_scorecard_v3/README.md).

Bộ slide này là bản đã dựng ngày 02/10/2026. Các file PowerPoint, PDF và
biểu đồ giữ nguyên bytes của bản đó. `manifest.json` ghi hash file và nguồn
để kiểm tra; không có ảnh render từng trang hoặc log build trong bộ commit.

Slide 12 ghi trạng thái đóng gói tại thời điểm dựng slide: chưa xác nhận
clean-clone. Sau đó, nhánh `benchmark-review` đã được clone từ GitHub và tái
dựng bảng thành công. Trạng thái mới ở
[báo cáo tái lập](../docs/reproduction_validation.md). Ghi chú cũ này không
thay đổi số liệu hay giới hạn khoa học của benchmark. Đây vẫn là benchmark
chưa đủ toàn bộ phép so sánh simulator đã dự kiến.

## Tái dựng biểu đồ

Script chỉ đọc các bảng scorecard đã commit, không đo simulator hoặc train.
Chạy từ root repository, với Python 3.10:

```bash
python3.10 -m venv .venv-presentation
.venv-presentation/bin/python -m pip install -r presentations/requirements-figures.txt
.venv-presentation/bin/python presentations/source/plot_report_figures.py
```

Kết quả nằm ở `work/presentation-figures/`. Script yêu cầu thư mục mới để
không ghi đè file có sẵn. Hash PNG/SVG có thể phụ thuộc phiên bản render/font;
nguồn bảng, số liệu, đơn vị và phạm vi so sánh phải giữ nguyên.

## Source của PowerPoint

`source/build_deck.mjs` là bản source đã điều chỉnh đường dẫn của generator
gốc. Nó cần runtime `@oai/artifact-tool`, font Noto Sans và helper trong kỹ
năng Presentations đã dùng khi dựng slide. Đặt `SKILL_DIR` tới thư mục kỹ năng
và `RUNTIME_PYTHON` tới Python của runtime đó, rồi chạy:

```bash
node presentations/source/build_deck.mjs
```

Generator ghi vào `work/presentation-deck/`, không ghi đè PowerPoint đã commit.
Bản xuất gốc đã được kiểm tra cấu trúc và render PDF, nhưng chưa kiểm tra mở
bằng Microsoft PowerPoint. Không hứa tái dựng PPTX byte-identical hoặc cài
runtime proprietary chỉ bằng `pip install`. Người không có runtime có thể
sửa trực tiếp file PowerPoint và dùng các biểu đồ đã cung cấp.

Các đường dẫn `work/repo_finalization/report_package/figures/` trong notes
của bản xuất cũ tương ứng với `presentations/figures/` trong nhánh này.
`work/repo_finalization/report_package/README.md` tương ứng với README này và
hướng dẫn tái lập ở `docs/`. Source tables nằm nguyên đường dẫn `artifacts/`.
