# Thực nghiệm Ultralytics YOLO

Mô hình YOLO11n khởi tạo từ trọng số COCO chính thức, fine-tune trực tiếp 10 lớp chữ số 0–9. Không dùng COCO để dự đoán chữ số khi chưa fine-tune.

Môi trường: Python 3.11.9, Ultralytics 8.3.221, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128; GPU NVIDIA RTX A2000 8GB Laptop. `pip check` đạt.

Dữ liệu: 1.200 ảnh train, 200 validation và 100 test tổng hợp ở kích thước 416×416. Nét chữ từ MNIST; chia train/val phân tầng seed42 và giữ nguồn MNIST test riêng. Seed tạo ảnh 20261031. Nền giấy gradient có nhiễu nhẹ, 1–10 chữ số/ảnh dương, 4% ảnh âm. Chỉ là dữ liệu tổng hợp, không phải ảnh chụp giấy thật. Script tạo box theo vị trí ghép, không dùng box do detector cũ dự đoán.

Cấu hình: 20 epochs, AdamW LR0 0.001, batch16, imgsz416, seed42, workers0, AMP tắt, không lật ảnh, xoay tối đa8°, mosaic0.3 và tắt mosaic5 epochs cuối. Checkpoint tốt nhất chọn bằng validation; test đánh giá sau huấn luyện. Cấu hình đầy đủ, CSV, loss curves và checkpoint nằm trong `models/yolo/<run>/`.

`summary.json` theo run chứa mAP50, mAP50–95, precision/recall trên test tổng hợp. Các chỉ số detection này không tương đương accuracy99,16% của CNN phân loại MNIST. Giao diện hiển thị confidence của box/lớp đã chọn, không suy diễn bảng xác suất10 lớp.

Nguồn: [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), [Train](https://docs.ultralytics.com/modes/train/).
