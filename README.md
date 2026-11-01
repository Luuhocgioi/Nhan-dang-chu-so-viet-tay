# Nhận dạng chữ số viết tay trên giấy

## Gói nộp bài

Dùng **[dist/DU_AN_NHAN_DANG_CHU_SO.zip](dist/DU_AN_NHAN_DANG_CHU_SO.zip)** để gửi mã nguồn và mô hình. Báo cáo chính: [reports/FINAL_REPORT.md](reports/FINAL_REPORT.md). Gói ZIP đã loại môi trường ảo, dataset thô và cache; cài lại thư viện theo hướng dẫn phía dưới khi giải nén trên máy khác.

Thư mục làm việc trên máy này vẫn giữ .venv, dữ liệu đang dùng và toàn bộ model cần thiết. Chi tiết dọn dẹp: [reports/CLEANUP.md](reports/CLEANUP.md).


Ứng dụng Python nhận ảnh chụp giấy, tìm vùng ký tự và hiển thị box xanh/nhãn vàng cho các chữ số0–9. Có CNN MNIST baseline, Ultralytics YOLO và bộ phân biệt chữ–số EMNIST62 lớp. Hỗ trợ ảnh có nhiều số rời nhau, giữ số0 đầu chuỗi, xoay ảnh, chọn vùng và xuất PNG/JSON.

**Chất lượng:** xem `reports/FINAL_REPORT.md` và `reports/real_photo_regression.json` để biết kết quả thực đo. Accuracy MNIST/mAP ảnh ghép không phải accuracy ảnh giấy thật. Chưa có bảo đảm bỏ mọi chữ cái, nhất là O/0 và I/l/1. Các ảnh do người dùng cung cấp là ảnh phát triển, không phải test độc lập.

## Mở ứng dụng

Trên máy hiện tại nhấp đúp `run_demo.cmd`. Chọn **Mở ảnh giấy…**, dùng ↶/↷ để các chữ số đứng đúng chiều, bấm **Nhận dạng**. **Không cần chọn phương pháp mỗi lần mở ảnh.** Ứng dụng dùng sẵn cấu hình mặc định. Luồng hằng ngày là Mở ảnh → xoay nếu cần → Nhận dạng → Lưu kết quả.

Các lựa chọn so sánh nằm trong **Nâng cao**, mặc định thu gọn. Đóng Nâng cao sẽ đặt lại phương pháp mặc định và tắt bộ lọc thử nghiệm. Danh sách thử nghiệm gồm:

- **Mặc định / Dãy số rời:** ít nhất3 ký tự cùng cỡ và cách nhau rộng; đã đọc đúng hai ảnh phát triển. Đây là giả định hàng số, chữ viết cách xa tương tự vẫn có thể bị nhận nhầm.
- **EMNIST chữ/số:** phân vùng nét và CNN62 lớp, chỉ xuất số đủ điểm; chữ và nét chưa đủ điểm nằm trong mục rejected của JSON.
- **YOLO EMNIST:** detector62 lớp từ ảnh ghép chữ viết tay; chỉ xuất lớp0–9.
- **YOLO:** baseline10 lớp đã khôi phục.
- **CNN:** phân vùng OpenCV + CNN10 lớp, có thể gán chữ thành số.

Trong Nâng cao, checkbox **Lọc mực xanh/tím** bật phương pháp chuyên màu cũ và ghi đè lựa chọn trong danh sách. Chỉ dùng khi cần so sánh trên giấy ô ly; chế độ này không phân biệt được chữ cùng màu. Ảnh mực đen không dùng checkbox này. Ảnh HEIC cần đổi sang JPG/PNG.

Kéo chuột chọn vùng giấy khi cần loại bàn/đồ vật. **Chọn toàn ảnh** để đặt lại. **Lưu kết quả…** xuất PNG có box/nhãn và JSON; nếu chọn ROI, xuất riêng vùng đó. Các bbox dùng tọa độ ảnh xuất, có thể khác ảnh gốc do resize. Với EMNIST, ảnh chỉ có chữ có thể có0 box; không ép dự đoán một số.

## Cài đặt (Python3.11 64-bit)

Chạy từ thư mục gốc trong PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-yolo.txt
.\.venv\Scripts\python.exe -m pip check
```

Trên máy có NVIDIA, cài Torch CUDA trước dependencies nếu cần GPU:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-yolo.txt
```

Máy kiểm tra: Windows, Python3.11.9, RTX A2000 8GB, Torch2.7.1+cu128, Ultralytics8.3.221, TensorFlow2.20.0. Tkinter đi kèm Python Windows. Không cần kích hoạt venv/đổi execution policy. Bản CPU vẫn chạy suy luận, train sẽ chậm hơn. `requirements-yolo-lock.txt` ghi môi trường đầy đủ; bản CUDA trong lock cần index PyTorch tương ứng.

## Chạy bằng dòng lệnh

```powershell
.\.venv\Scripts\python.exe -m src.demo_photo
.\.venv\Scripts\python.exe -m src.detect "data/samples/mnist_paper_row.png" --backend yolo --output reports/figures/result.png
.\.venv\Scripts\python.exe -m src.detect "anh_giay.jpg" --backend emnist --output reports/figures/result_emnist.png
.\.venv\Scripts\python.exe -m src.detect "anh_giay.jpg" --backend yolo-emnist --output reports/figures/result_yolo_emnist.png
.\.venv\Scripts\python.exe -m src.check_real_photos
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`check_real_photos` cần5 ảnh gốc trong Downloads, hoặc truyền `--downloads <thu_muc>`. Thiếu ảnh được ghi là missing_input, không tự coi là đạt. Tên ảnh và hướng xoay nằm trong mã kiểm tra; đây là kiểm tra phát triển, không dùng để báo cáo accuracy tổng quát.

## Huấn luyện và tái lập

CNN MNIST:

```powershell
.\.venv\Scripts\python.exe -m src.data
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m src.evaluate
```

EMNIST chính thức (lần tải đầu cần mạng):

```powershell
.\.venv\Scripts\python.exe -c "from torchvision.datasets import EMNIST; EMNIST('data/raw/emnist',split='byclass',train=True,download=True)"
.\.venv\Scripts\python.exe -m src.train_emnist
.\.venv\Scripts\python.exe -m src.yolo_emnist_dataset
.\.venv\Scripts\python.exe -m src.train_yolo_emnist
```

EMNIST classifier giữ62 lớp khi train, chỉ xuất0–9 khi suy luận. Chuẩn hóa gồm transpose đúng hướng, crop, resize giữ tỉ lệ20px, padding28px. Train/validation tách từ train chính thức; test chính thức chỉ đo sau chọn checkpoint. Các split chưa được xác nhận độc lập theo người viết nên không khẳng định writer-independent.

YOLO EMNIST dùng ảnh ghép nhiều ký tự, có mực nhạt/nền gradient/ô ly; ký tự là chữ viết tay thật, toàn cảnh vẫn là tổng hợp. YAML có62 lớp; class>=10 bị loại khi xuất. Giữ pretrained baseline cho so sánh, không dùng COCO trực tiếp để nhận chữ số.

Các script EMNIST lưu con trỏ candidate, không tự thay YOLO mặc định. Không điều chỉnh mô hình theo test chính thức. Khi tiếp tục huấn luyện cần so sánh validation và ảnh phát triển, ghi rõ hồi quy/bỏ sót, rồi mới chốt phương pháp phát hành.

## Sản phẩm

| Đường dẫn | Nội dung |
|---|---|
| `src/` | Mã tải, train, đánh giá, tiền xử lý, nhận dạng và GUI |
| `tests/` | Kiểm thử tiền xử lý, dữ liệu, inference và GUI |
| `models/*.json` | Con trỏ checkpoint baseline/candidate/runtime |
| `models/<run>/` | Checkpoint classifier và cấu hình |
| `models/yolo/<run>/` | YOLO weights, loss curves, CSV, args.yaml |
| `reports/FINAL_REPORT.md` | Báo cáo hiện tại, kết quả thực đo và giới hạn |
| `reports/experiments/` | Metrics/history/seed/split theo run |
| `reports/figures/` | Minh chứng ảnh đầu vào, box, lỗi và biểu đồ |
| `reports/report.md` | Nhật ký báo cáo các thí nghiệm trước |
| `notebooks/` | Notebook khám phá MNIST tùy chọn; cần Jupyter riêng |
| `PROGRESS.md` | Trạng thái và công việc còn lại |

Đóng gói mã, tài liệu, checkpoint và minh chứng:

```powershell
.\.venv\Scripts\python.exe -m src.package_project
```

ZIP nằm trong `dist/`, không chứa venv hoặc dataset thô. Giải nén, cài dependencies rồi chạy `run_demo.cmd`. Đính kèm thông tin sinh viên/lớp và chuyển báo cáo sang định dạng giảng viên yêu cầu trước khi nộp. Chưa có video demo hoặc xác nhận nộp bài.

## Giới hạn và xử lý lỗi

- Ảnh chụp quá xa, xoay sai, mực nhạt, chữ/số dính nhau và đồ vật vẫn có thể gây sai. Kiểm tra vùng nét và chọn ROI trước.
- Lọc màu ưu tiên nét xanh/tím; ảnh có nhiều màu hoặc nền cùng màu cần kiểm tra thủ công. Nét đen dùng đường xử lý grayscale.
- Không có số đủ điểm không có nghĩa chắc chắn ảnh không có số. Confidence chưa hiệu chuẩn thành xác suất đúng.
- Thiếu checkpoint: kiểm tra JSON trong models hoặc train lại theo lệnh tương ứng. Cửa sổ đang mở giữ model trong RAM; mở lại sau thay con trỏ.
- Thiếu webcam không ảnh hưởng luồng ảnh giấy; canvas/webcam là demo phụ.

## Tài liệu tham khảo

- [NIST EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- [Torchvision EMNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.EMNIST.html)
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Keras MNIST](https://keras.io/api/datasets/mnist/)
- [DataFlair theo đề bài](https://data-flair.training/blogs/python-deep-learning-project-handwritten-digit-recognition/)

Đầu phiên Codex đọc PROGRESS.md trước rồi PROMPT.md và chỉ tiếp tục các phần chưa hoàn thành hoặc có lỗi mới.

Ảnh ghép nhiều ô sáng trên nền tối được tự phát hiện: hệ thống bỏ viền từng ô trước khi đọc nét số. Không cần chọn chế độ riêng. Đã kiểm tra image-15.png đọc50419/21314; xem reports/figures/image15_after.png. Đây là ca phát triển đã dùng sửa phân vùng.

Đã bổ sung tự phát hiện mực đỏ, giữ cả nét đen do viền/nén ảnh quanh nét đỏ. Ca OIP (1).webp đã đọc đúng bảng0–9; chữ và số trong tiêu đề không thuộc tiêu chí hồi quy bảng số. Bộ kiểm tra hiện gồm4 ảnh phát triển. Kết quả chưa đại diện mọi kiểu ảnh/độ chính xác tổng quát.

Ảnh lưới dày OIP.webp: đã tách200 vùng, đọc đúng199 theo các hàng0–9 (20 mẫu/hàng), còn4→8 ở hàng5 cột14. Không điền nhãn theo hàng để che lỗi. Xem reports/dense_evaluation.json. Đây là ca phát triển thứ5, chưa rõ nguồn gốc ảnh, không phải benchmark độc lập. Phần kiểm tra exact_match của ca này vẫn phải thất bại cho đến khi mô hình thực sự đọc đúng.
