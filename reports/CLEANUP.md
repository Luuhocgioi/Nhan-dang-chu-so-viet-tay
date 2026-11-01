# Dọn dự án để chuẩn bị nộp bài

Đã xóa143 file dư, tổng khoảng1.900MiB (chi tiết chính xác trong cleanup_result.json). Mục tiêu là bỏ bản trùng/cache và giữ nguyên chức năng hiện tại.

## Đã xóa

- Sáu gói ZIP phát hành cũ; thay bằng một gói nộp duy nhất tại `dist/DU_AN_NHAN_DANG_CHU_SO.zip`.
- Archive gzip.zip EMNIST đã giải nén và các split bymerge/balanced/letters/digits/mnist không được mã nguồn sử dụng. Giữ đủ4 file IDX của ByClass.
- Log huấn luyện/chẩn đoán tạm, bytecode Python, cache nhãn YOLO có thể tạo lại.
- Checkpoint last.pt cuối epoch không được con trỏ mô hình hoặc script hiện tại dùng; giữ toàn bộ best.pt/best.keras được tham chiếu và trọng số khởi tạo yolo11n.pt.
- Ảnh train_batch/val_batch tạm trong thư mục huấn luyện; giữ biểu đồ, cấu hình, CSV, metrics và minh chứng test cuối.
- Các mask, ảnh zoom và phép thử threshold trung gian không được tài liệu tham chiếu. Giữ ảnh trước/sau và ảnh lỗi cần phân tích.

## Đã giữ

- Toàn bộ src, tests, notebooks, requirements, PROMPT, PROGRESS, README và báo cáo.
- Toàn bộ model best đang dùng cho mặc định và các lựa chọn Nâng cao, cùng JSON con trỏ/cấu hình.
- .venv để chương trình trên máy này tiếp tục chạy; .git để giữ lịch sử; dữ liệu MNIST, EMNIST ByClass và các dataset YOLO mà script train/đánh giá đang dùng.
- Thống kê/curves/confusion matrix/split chỉ số, các ảnh minh chứng, lỗi chưa giải quyết và ảnh mẫu kiểm thử.
- Cấu hình Ultralytics hiện có để tránh thay đổi hành vi môi trường.

## Kiểm tra sau dọn

51 file mã nguồn/kiểm thử/requirements/mô hình được bảo vệ đã đối chiếu SHA256 không thay đổi. Các con trỏ checkpoint vẫn trỏ đến file tồn tại. EMNIST ByClass train697.932/test116.323 mẫu đọc được bằng download=False, không cần tải lại.

31 kiểm thử đều đạt. Bốn ảnh phát triển trước vẫn đúng chuỗi; ảnh lưới dày vẫn tách200 số và sai đúng1 số4→8 như trước, không phát sinh hồi quy. Đây không phải chứng nhận accuracy trên ảnh mới.

## File mang nộp

Dùng ZIP `dist/DU_AN_NHAN_DANG_CHU_SO.zip` và báo cáo `reports/FINAL_REPORT.md`. ZIP không chứa .venv, .git, dataset thô/cache tải về; có mã nguồn, requirements, checkpoint cần thiết, báo cáo và minh chứng. Người nhận giải nén, cài requirements theo README rồi chạy run_demo.cmd. Còn cần điền thông tin sinh viên/lớp và theo định dạng báo cáo của giảng viên.

Lưu ý: thư mục __pycache__ hoặc log có thể xuất hiện lại khi chạy; đó là file runtime tạo tự động. Chạy trên máy đang làm việc cần giữ .venv. Không gửi nguyên toàn bộ thư mục máy làm việc với venv/dataset.
