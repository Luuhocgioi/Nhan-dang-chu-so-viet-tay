# PROMPT — Hệ thống nhận dạng chữ số viết tay

## 1. Vai trò và mục tiêu

Bạn là kỹ sư AI/Computer Vision hỗ trợ thực hiện đề tài cuối kỳ môn **Xử lý ảnh & Nhận dạng**. Hãy xây dựng hệ thống nhận dạng chữ số viết tay bằng Python, giải thích được các bước xử lý ảnh, kiến trúc CNN và kết quả thực nghiệm. Ưu tiên mã nguồn rõ ràng, chạy lại được, phù hợp máy cá nhân và có demo dễ sử dụng.

**Trước mỗi phiên làm việc, đọc `PROGRESS.md` trước, sau đó đọc toàn bộ `PROMPT.md`.** Kiểm tra mã nguồn và các sản phẩm hiện có trước khi quyết định bước tiếp theo.

## 2. Phạm vi và đầu vào/đầu ra

- Tự phát hiện **nhiều chữ số từ 0 đến 9** viết rời nhau trên giấy; tách từng vùng, phân loại bằng CNN và ghép kết quả theo dòng, từ trái sang phải. Giữ cả số 0 ở đầu chuỗi.
- Đầu vào huấn luyện: ảnh và nhãn của MNIST.
- Đầu vào chính (đã làm rõ với người dùng): **ảnh chụp chữ số viết tay trên giấy**, tải từ máy tính. Người dùng viết trên giấy rồi chụp ảnh, không cần vẽ trên máy tính. Tự khoanh từng chữ số bằng khung xanh, ghi nhãn vàng phía trên như mẫu người dùng. Hiển thị chuỗi dự đoán; cho phép xoay, chọn vùng giấy tùy chọn và lưu ảnh kết quả/JSON.
- Đầu ra: nhãn dự đoán, điểm softmax của các lớp và ảnh sau tiền xử lý để người dùng kiểm tra. Điểm softmax không được diễn giải như bảo đảm dự đoán đúng.
- Khi ảnh trống hoặc không tìm thấy nét chữ, hiển thị thông báo phù hợp thay vì ép dự đoán một chữ số.
- Đã mở rộng theo mẫu người dùng ngày 2026-10-30: nhận dạng chuỗi nhiều chữ số rời nhau. Chưa hỗ trợ OCR chữ cái/đoạn văn, chữ số dính nhau hoặc tách/gộp nét đứt một cách tin cậy. Ghi rõ giới hạn và đánh giá ảnh giấy thật.
- Ưu tiên hoàn thiện luồng ảnh giấy và đánh giá ảnh chụp thật. Canvas/webcam là công cụ phụ đã có, không phải yêu cầu người dùng phải sử dụng.

## 3. Dữ liệu và tài liệu tham khảo

### 3.1. Nguồn tham khảo

- Bài tham khảo theo yêu cầu: [DataFlair — Handwritten Digit Recognition using Python](https://data-flair.training/blogs/python-deep-learning-project-handwritten-digit-recognition/).
- Nguồn xác nhận cấu trúc dữ liệu và API tải: [Keras — MNIST digits classification dataset](https://keras.io/api/datasets/mnist/).
- Tham khảo cách huấn luyện/đánh giá: [Keras — Training and evaluation with built-in methods](https://keras.io/guides/training_with_built_in_methods/).

Dùng bài DataFlair làm tài liệu định hướng và trích dẫn trong báo cáo; không sao chép máy móc mã hoặc phiên bản thư viện cũ. Ngày 2026-10-30, truy cập trực tiếp bài DataFlair trả HTTP 403; cần đọc lại khi truy cập được trước khi mô tả chi tiết triển khai của bài. MNIST dưới đây là lựa chọn chủ động của dự án, dựa trên tài liệu chính thức Keras.

### 3.2. Dataset dự kiến

- MNIST gồm 60.000 ảnh huấn luyện và 10.000 ảnh kiểm thử; ảnh grayscale 28 × 28, nhãn từ 0 đến 9. Tải bằng `tf.keras.datasets.mnist.load_data()` theo cấu trúc dữ liệu được mô tả trong tài liệu Keras nêu trên.
- Tách 6.000 ảnh từ tập huấn luyện làm validation, ưu tiên phân tầng theo nhãn và seed cố định; còn 54.000 ảnh để train.
- Giữ nguyên tập test 10.000 ảnh. Không dùng test để chọn kiến trúc, epoch, ngưỡng hoặc siêu tham số.
- Khám phá số mẫu, kích thước, kiểu dữ liệu, phân bố lớp và ảnh minh họa trước khi huấn luyện.
- Tạo bộ ảnh thực tế tự viết có nhãn để kiểm tra demo; ghi nguồn, số lượng, điều kiện chụp và kết quả riêng, không gộp với MNIST test.
- Không đưa dữ liệu tải về hoặc cache lớn vào Git. Ghi cách tải lại và vị trí lưu trong README.

## 4. Yêu cầu kỹ thuật

### 4.1. Công nghệ

- Ngôn ngữ: Python; tạo môi trường ảo `.venv`.
- Theo yêu cầu ngày 2026-10-31, tích hợp Ultralytics YOLO11n (PyTorch) để dự đoán trực tiếp box và lớp chữ số; giữ TensorFlow/Keras CNN làm baseline so sánh. Mô hình COCO phải fine-tune 10 lớp 0–9 trước khi dùng. Backend auto ưu tiên checkpoint YOLO đã huấn luyện.
- Thư viện: OpenCV, NumPy, Matplotlib, scikit-learn để chia dữ liệu và tính metrics; Jupyter nếu cần notebook khám phá.
- Giao diện chính dùng Tkinter (`src/demo_photo.py`) để chọn ảnh giấy, tự tìm nhiều chữ số, xoay và nhận dạng. OpenCV xử lý ảnh; Pillow đọc EXIF điện thoại. Canvas/webcam chỉ là chế độ phụ.
- Xác minh phiên bản Python/framework tương thích trước khi cài; ghi phiên bản thực tế đã kiểm tra vào `requirements.txt` và README. CPU là cấu hình tối thiểu, GPU là tùy chọn.

### 4.2. Pipeline xử lý ảnh

Luồng tổng thể: **dữ liệu → tiền xử lý → huấn luyện CNN → đánh giá → nạp mô hình → dự đoán ảnh/webcam/canvas**.

Với ảnh thực tế:

1. Đọc ảnh, xác minh ảnh hợp lệ; sửa hướng theo EXIF, xử lý kênh alpha nếu có và chuyển grayscale. Với ảnh giấy, ước lượng nền bằng morphological closing và chia ảnh cho nền để giảm ảnh hưởng ánh sáng không đều; loại thành phần chạm mép ROI (phải chừa viền quanh chữ số).
2. Khử nhiễu nhẹ khi cần. Dùng threshold (Otsu hoặc adaptive tùy điều kiện) để xác định vùng nét chữ; thống nhất nét sáng trên nền tối như MNIST.
3. Loại vùng nhiễu nhỏ, lấy bounding box của chữ số trong ROI; giữ các thành phần thuộc cùng chữ số. Phát hiện ảnh trống.
4. Crop, resize giữ tỉ lệ để chữ số nằm trong vùng khoảng 20 × 20, thêm padding thành 28 × 28 và căn giữa; tránh kéo méo nét.
5. Chuyển `float32`, chuẩn hóa về `[0, 1]`, thêm chiều kênh và batch theo framework. Với TensorFlow, tensor dự đoán có dạng `(N, 28, 28, 1)`.

MNIST đã là grayscale 28 × 28 nên không cần threshold hoặc crop lại một cách máy móc. Threshold ảnh thực tế có thể chỉ dùng để tìm bounding box, giữ ảnh grayscale khi đưa vào CNN. Thử nghiệm lựa chọn này trên validation/bộ ảnh phát triển và lưu cấu hình. Dùng chung hàm tiền xử lý cho ảnh, webcam và canvas; thống nhất cực tính, miền giá trị và thứ tự kênh với dữ liệu train.

Nếu dùng augmentation, chỉ áp dụng trên train với phép dịch/xoay nhỏ hợp lý. Không lật ảnh làm thay đổi ý nghĩa chữ số; không augment validation/test.

### 4.3. CNN và huấn luyện

- Baseline đề xuất: `Input(28,28,1) → Conv2D(32,3,relu,padding=same) → MaxPool2D → Conv2D(64,3,relu,padding=same) → MaxPool2D → Flatten → Dense(128,relu) → Dropout(0.3) → Dense(10,softmax)`.
- Cấu hình khởi đầu: Adam, learning rate `0.001`, batch size `128`, tối đa `15` epochs, `sparse_categorical_crossentropy` cho nhãn số nguyên, metric `accuracy`.
- Cố định seed, ví dụ `42`; ghi cấu hình, phiên bản thư viện, thiết bị và thời gian huấn luyện. Nêu giới hạn tính tái lập nếu phần cứng/backend không hoàn toàn xác định.
- Theo dõi validation loss, dùng EarlyStopping và lưu checkpoint tốt nhất. Lưu mô hình Keras dạng `.keras`, history và cấu hình của từng lần chạy vào thư mục có mã run riêng.
- Không ghi đè kết quả cũ khi chưa lưu đủ bằng chứng. Kiểm tra mô hình nạp lại dự đoán nhất quán với mô hình trước khi lưu trên cùng mẫu.

### 4.4. Đánh giá

- Báo cáo test loss và test accuracy của mô hình đã chọn bằng validation.
- Xuất confusion matrix đủ 10 lớp, ghi rõ hàng là nhãn thật, cột là nhãn dự đoán; bổ sung precision/recall/F1 theo lớp khi hữu ích.
- Vẽ loss curves và accuracy curves của train/validation theo epoch; lưu hình và số liệu gốc.
- Phân tích một số mẫu dự đoán sai, cặp lớp dễ nhầm và dấu hiệu overfitting.
- Đánh giá ảnh thực tế riêng: số mẫu, độ chính xác nếu có nhãn, ví dụ thành công/thất bại và hạn chế ánh sáng, nét chữ, nền ảnh.
- Mục tiêu định hướng: test accuracy MNIST ≥ 98%. Đây là mục tiêu, không phải kết quả đã đạt. Nếu chưa đạt, ghi kết quả thật và phân tích; không điều chỉnh dựa liên tục trên test.

## 5. Cấu trúc thư mục dự kiến

```text
Nhan-dang-chu-so-viet-tay/
├── PROMPT.md
├── PROGRESS.md
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                   # Dữ liệu/cache gốc
│   ├── processed/             # Dữ liệu đã xử lý nếu cần lưu
│   └── samples/               # Ảnh tự viết và nhãn kiểm tra demo
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── detect.py              # Tách nhiều chữ số, sắp thứ tự, annotate và xuất JSON
│   ├── predict.py
│   ├── demo_photo.py          # Giao diện chính nhận ảnh giấy
│   ├── demo_canvas.py
│   └── demo_webcam.py
├── models/
│   └── <run_id>/              # Checkpoint và cấu hình từng lần chạy
├── notebooks/
│   └── 01_explore_mnist.ipynb
└── reports/
    ├── figures/
    ├── experiments/           # History, metrics, cấu hình, thông tin run
    └── report.md
```

Đây là cấu trúc dự kiến, không hàm ý các file đã tồn tại. Notebook dùng để khám phá/minh họa; chức năng chính phải chạy được từ `src/`. README phải có lệnh cài đặt, tải dữ liệu, train, evaluate và chạy từng demo đúng với mã đã triển khai.

## 6. Tiêu chí hoàn thành và sản phẩm nộp

- [ ] Môi trường cài đặt được từ hướng dẫn; có phiên bản phụ thuộc và cách chạy trên CPU.
- [ ] Dữ liệu được khám phá, chia train/validation/test đúng; pipeline tiền xử lý được mô tả và có minh họa.
- [ ] CNN huấn luyện được; checkpoint, cấu hình, seed và history được lưu, mô hình nạp lại được.
- [ ] Có accuracy, loss, confusion matrix, curves và phân tích lỗi từ lần chạy thật; nêu rõ tình trạng đạt mục tiêu accuracy.
- [ ] Giao diện ảnh giấy hoạt động: mở ảnh, tự tìm nhiều chữ số, khung xanh/nhãn vàng, chuỗi kết quả, xuất PNG/JSON, chọn ROI tùy chọn, xoay; xử lý ảnh trống và file lỗi. Đánh giá bằng ảnh giấy thật có nhãn; canvas/webcam là bổ sung.
- [ ] README hướng dẫn người khác tái hiện; báo cáo giải thích dữ liệu, phương pháp, thực nghiệm, giới hạn và tài liệu tham khảo.
- [ ] Gói nộp gồm mã nguồn, requirements, mô hình hoặc hướng dẫn lấy mô hình, báo cáo theo định dạng môn học yêu cầu, ảnh biểu đồ và video/kịch bản demo.
- [ ] Chạy thử quy trình theo README, rà soát file nộp và cập nhật trạng thái cuối trong `PROGRESS.md`.

Không đánh dấu hoàn thành một mục chưa có bằng chứng. Nếu chưa biết yêu cầu định dạng báo cáo, hạn nộp hoặc quy định đóng gói, ghi vào vấn đề mở; vẫn tiếp tục các việc độc lập.

## 7. Quy ước làm việc hằng ngày với Codex

1. **Đầu phiên:** đọc `PROGRESS.md` trước rồi `PROMPT.md`; kiểm tra các file liên quan. Tóm tắt phần đã làm, việc còn thiếu và bước tiếp theo.
2. **Trong phiên:** thực hiện công việc theo Next steps, ưu tiên phần nhỏ có thể kiểm chứng. Giải thích quyết định kỹ thuật cần thiết để sinh viên hiểu và bảo vệ đề tài.
3. **Sau mỗi bước có ý nghĩa:** cập nhật ngày sửa, checklist, nhật ký, kết quả thực nghiệm nếu có và Next steps trong `PROGRESS.md`. Ghi đường dẫn sản phẩm và cách kiểm tra.
4. **Cuối phiên:** ghi rõ việc đang dở, lệnh tiếp tục nếu đã xác minh, lỗi còn tồn tại và bước đầu tiên của phiên sau.
5. Không làm lại phần đã hoàn thành trừ khi cần sửa lỗi hoặc yêu cầu thay đổi. Nếu tiến trình và mã nguồn mâu thuẫn, kiểm tra bằng chứng rồi sửa trạng thái; không mặc định checklist là đúng.
6. Không bịa accuracy, thời gian chạy hoặc kết quả kiểm thử. Phân biệt rõ **dự kiến**, **đã triển khai**, **đã chạy**, **đã xác minh**.
7. Không tự xóa dữ liệu, ghi đè mô hình tốt hoặc thay đổi ngoài phạm vi. Giữ mã đơn giản, tách trách nhiệm và tránh đường dẫn tuyệt đối phụ thuộc máy cá nhân.

Mẫu yêu cầu đầu phiên:

> Hãy đọc PROGRESS.md trước rồi PROMPT.md, kiểm tra trạng thái thực tế, tóm tắt tiến độ và thực hiện bước tiếp theo phù hợp. Sau mỗi bước, cập nhật PROGRESS.md với kết quả đã xác minh, vấn đề còn mở và Next steps. Không làm lại những phần đã hoàn thành nếu không có lý do.

## Bổ sung YOLO (2026-10-31)

Dùng `src/yolo_dataset.py` tạo dữ liệu detection tổng hợp có box, `src/train_yolo.py` train/đánh giá, `src/yolo_predict.py` suy luận. Lưu cấu hình, checkpoint, mAP50/mAP50–95, precision/recall. Không so sánh mAP detection trực tiếp với accuracy phân loại MNIST. Bắt buộc ghi rõ ảnh tổng hợp và ảnh chụp thật; không gán kết quả tổng hợp thành độ chính xác thực tế.

Yêu cầu2026-11-01: ảnh có cả chữ và số, chỉ xuất box số. Cho phép lớp10 other_letter khi huấn luyện và loại lớp này lúc xuất kết quả. Bắt buộc đo chữ→số, precision và recall số trên cùng bộ ảnh hỗn hợp, nêu rõ tradeoff bỏ sót và nguồn dữ liệu tổng hợp/thật.

Quyết định2026-11-01 sau phản hồi: mặc định giữ digits_20261031_180956; mixed là candidate. Ưu tiên giữ chất lượng nhận số; đánh giá ảnh giấy thật và kiểm tra hồi quy trước khi thay mô hình mặc định, không dùng riêng kết quả chữ font tổng hợp để quyết định.

Bổ sung chế độ blue-ink tùy chọn cho giấy ô ly/mực xanh tím; không thay default YOLO, không quảng bá như loại chữ tổng quát. Ảnh maxresdefault (1).jpg là ảnh phát triển đã dùng điều chỉnh bộ lọc, không dùng làm test độc lập.

Bản phát hành2026-11-01: models/runtime.json chọn numeric-rows theo hai ảnh phát triển; điều kiện≥3 ký tự cùng cỡ, cách nhau rộng phải được hiển thị rõ. Đây là giả định bố cục, không được quảng bá như loại chữ tổng quát. CNN62 và YOLO62 đã train/đánh giá riêng; giữ các lựa chọn để so sánh. Báo cáo mới tại reports/FINAL_REPORT.md. Chưa có benchmark ảnh thật độc lập, không thay các giới hạn bằng lời hứa100%.

Quy ước giao diện2026-11-01: không bắt người dùng chọn model/phương pháp theo ảnh. Luồng chính Mở ảnh → xoay nếu cần → Nhận dạng. Các lựa chọn nghiên cứu nằm trong Nâng cao thu gọn; đóng mục này khôi phục mặc định. Không diễn giải cấu hình mặc định như tự tìm mô hình tối ưu cho mọi ảnh.
