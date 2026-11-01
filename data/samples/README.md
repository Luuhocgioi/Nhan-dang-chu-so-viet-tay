# Ảnh kiểm tra

`synthetic_1.png` là nét thẳng do OpenCV tạo trên nền trắng, nhãn kỳ vọng 1. Ảnh này chỉ kiểm tra luồng dự đoán, không phải chữ viết tay thật và không được dùng để báo cáo độ chính xác thực tế.

Thêm ảnh viết tay độc lập vào đây khi thu thập được; ghi nhãn, nguồn, điều kiện chụp và mục đích (phát triển hoặc đánh giá). Không đưa ảnh đánh giá vào dữ liệu dùng điều chỉnh tiền xử lý.

`mnist_paper_row.png`: ghép các mẫu đầu tiên của từng lớp 0,5,2,2,1,8,5,9 trong train split seed 42, phóng 84×84, đặt trên nền gradient 160–240. Chỉ là kiểm tra luồng nhiều số, có dữ liệu từ train; không được dùng báo cáo accuracy độc lập.
