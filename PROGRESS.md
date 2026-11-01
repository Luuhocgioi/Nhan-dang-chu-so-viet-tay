# PROGRESS — Theo dõi dự án nhận dạng chữ số viết tay

## 1. Thông tin chung

| Thuộc tính | Giá trị |
| --- | --- |
| Tên đề tài | Xây dựng hệ thống nhận dạng chữ số viết tay |
| Môn học | Xử lý ảnh & Nhận dạng |
| Ngày bắt đầu ghi nhận | 2026-10-30 |
| Ngày cập nhật gần nhất | 2026-11-01 |
| Múi giờ | Asia/Bangkok (UTC+7) |
| Trạng thái | Bản chạy Dãy số rời đã kiểm tra hai ảnh phát triển; CNN62/YOLO62 đã train, báo cáo và gói sản phẩm đã hoàn tất; chưa có benchmark ảnh thật độc lập |
| Dataset dự kiến | MNIST; bổ sung ảnh tự viết để kiểm tra thực tế |
| Công nghệ dự kiến | Python, TensorFlow/Keras, OpenCV, NumPy, Matplotlib, scikit-learn; Tkinter cho canvas |
| Hạn nộp / thành viên | Chưa cung cấp |

Ngày bắt đầu trên là ngày tạo hồ sơ tiến trình, không khẳng định đây là ngày khởi tạo repository. Khi kiểm tra ban đầu, thư mục có `.git/` và `README.md` chỉ chứa tiêu đề dự án. Đây là trạng thái ban đầu; các mục bên dưới ghi kết quả triển khai mới nhất.

## 2. Checklist các giai đoạn

- [x] **Thiết lập môi trường & cấu trúc thư mục:** Python 3.11, `.venv`, requirements và lock, `pip check` đạt, Tkinter 8.6 khởi tạo thành công.
- [x] **Tải và khám phá dữ liệu:** đã tải MNIST, xuất `reports/data_summary.json`, ảnh minh họa và phân bố lớp; chia 54.000/6.000/10.000.
- [x] **Tiền xử lý dữ liệu:** đã triển khai; 6 kiểm thử về ảnh trống, cực tính, dịch chuyển, alpha, nhiễu và file lỗi đạt. Chưa đánh giá trên bộ chữ viết thật có nhãn.
- [x] **Xây dựng mô hình CNN:** đã xây baseline và đang huấn luyện; đầu ra 10 lớp, model summary lưu theo run.
- [x] **Huấn luyện & lưu mô hình:** chạy train, lưu checkpoint tốt nhất, history, seed và cấu hình; xác minh nạp lại.
- [x] **Đánh giá mô hình:** lưu test metrics, confusion matrix, curves và phân tích mẫu sai.
- [x] **Chức năng nhận dạng ảnh giấy:** mở/xoay hai hướng/ROI/box/nhãn/chuỗi/xuấtPNG-JSON, lựa chọn phương pháp, trạng thái không có số; đã kiểm tra hai ảnh thật phát triển và GUI. Chất lượng tổng quát chưa được xác nhận; Dãy số rời có giả định bố cục. Canvas/webcam là phụ.
- [x] **Viết báo cáo & README:** trình bày phương pháp, kết quả thật, hướng dẫn chạy, giới hạn và nguồn tham khảo.
- [ ] **Hoàn thiện & nộp bài:** chạy lại theo README, rà soát gói nộp, chuẩn bị demo và xác nhận việc nộp thực tế.

Chỉ đổi `[ ]` thành `[x]` khi đủ bằng chứng cho toàn bộ giai đoạn. Nếu hoàn thành một phần, ghi phần đó vào nhật ký; không đánh dấu cả giai đoạn.

## 3. Nhật ký công việc

| Ngày | Công việc đã làm | Kết quả / bằng chứng | Vấn đề gặp phải |
| --- | --- | --- | --- |
| 2026-10-30 | Kiểm tra thư mục và chuẩn bị tài liệu quản lý dự án | Tạo `PROMPT.md` và `PROGRESS.md`; ghi phạm vi, pipeline, tiêu chí hoàn thành và công việc tiếp theo. README hiện mới có tiêu đề. | Chưa triển khai hay chạy thí nghiệm. Truy cập trực tiếp bài DataFlair trả HTTP 403; thông tin MNIST được đối chiếu với tài liệu Keras. |
| 2026-10-30 | Tạo `.venv` bằng Python 3.11, cấu trúc thư mục và các module dữ liệu/CNN/train/evaluate/predict/demo | Đã viết mã trong `src/` và kiểm thử tiền xử lý trong `tests/`; đang cài requirements | Lần tải pip trong sandbox lỗi DNS; tải lại với quyền mạng đã bắt đầu thành công. Chưa xác minh runtime. |
| 2026-10-30 | Cài xong thư viện, tải/khám phá MNIST, chạy kiểm thử và bắt đầu train | `pip check` đạt; 6/6 unit tests đạt; Tkinter 8.6 hoạt động; xuất hình dữ liệu và bắt đầu `python -m src.train` | TensorFlow ghi cảnh báo thuộc tính dataset không nhận biết, sau đó tiếp tục huấn luyện bình thường. |
| 2026-10-30 | Huấn luyện và đánh giá baseline | 11 epochs; best epoch 8; reload nhất quán; test 99,16%, loss 0,026190; xuất confusion matrix/curves/errors | Lỗi cp1252 ở dòng in cuối sau khi đã lưu đủ sản phẩm; đã sửa UTF-8 cho stdout/stderr, không cần train lại |
| 2026-10-30 | Kiểm tra demo và viết tài liệu | 9/9 tests đạt; CLI ảnh tổng hợp dự đoán đúng lớp 1; hoàn thành README, notebook khám phá và `reports/report.md` | Camera index 0 không mở được; chưa có tập chữ viết thật, notebook chưa chạy bằng Jupyter |
| 2026-10-30 | Người dùng làm rõ đầu vào là chữ số viết trên giấy | Tạo demo_photo; đổi run_demo.cmd sang tải ảnh; hỗ trợ EXIF, xoay, ROI, giảm nền không đều. 13/13 tests đạt; CLI paper chạy được | Mẫu kiểm thử tổng hợp; chưa đo accuracy ảnh giấy thật |

| 2026-10-30 | Mở rộng theo hình mẫu: nhiều chữ số, khung xanh, nhãn vàng | Thêm src/detect.py; cập nhật GUI tự tách số, xuất chuỗi/PNG/JSON. 17/17 tests đạt; ảnh MNIST train ghép nền gradient đọc 05221859 (8 vùng) | Ảnh mẫu người dùng đã có overlay nên chỉ làm tham chiếu hiển thị. Chưa đo ảnh giấy thật; chưa hỗ trợ đáng tin cậy chữ dính/nét đứt |

| 2026-10-31 | Tích hợp Ultralytics theo yêu cầu | Phát hiện GPU RTX A2000 8GB; đang tải PyTorch CUDA. Tạo 1.200 train/200 val/100 test ảnh ghép MNIST có box và 4% ảnh âm; tách nguồn MNIST độc lập. Thêm train_yolo/yolo_predict, GUI và CLI hỗ trợ backend auto/yolo/cnn | Đã hoàn thành cài đặt, train 20 epochs (~447,5 giây); đánh giá test tổng hợp 100 ảnh/557 số. mAP50 99,2604%, mAP50–95 97,8113%, precision 99,2441%, recall 98,5015% ở imgsz416 |

| 2026-10-31 | Chạy thử YOLO và tích hợp GUI | Checkpoint models/yolo/digits_20261031_180956/weights/best.pt; ảnh demo có 8 box, chuỗi 05221859; 20 tests đạt trước đổi kích thước, 3 tests YOLO chạy lại đạt sau đổi | Ảnh ngang ở imgsz416 từng nhầm 8 thành9; đặt suy luận640 giữ nét tốt hơn, demo đọc đúng. Chưa có đánh giá ảnh giấy thật; mAP test báo cáo vẫn ở416 |

| 2026-10-31 | Mở và kiểm tra ứng dụng thật | Cửa sổ “Đọc chữ số viết tay trên giấy — Ultralytics YOLO” hiển thị đúng8 box, nhãn vàng và chuỗi05221859. Giữ cửa sổ mở để người dùng chọn ảnh riêng | Ảnh demo là tổng hợp; chưa xác minh ảnh chụp giấy của người dùng |

Thêm một dòng sau mỗi bước có ý nghĩa; không xóa lịch sử. Ghi ngày dạng `YYYY-MM-DD`, sản phẩm đã tạo, lệnh hoặc cách kiểm tra thực tế và lỗi cụ thể nếu có. Nếu thay đổi quyết định cũ, thêm dòng giải thích lý do.

## 4. Kết quả thực nghiệm

**Đã hoàn thành một lần huấn luyện baseline.** Đơn vị accuracy trong bảng là phần trăm. Checkpoint chọn theo validation loss tại epoch 8; train accuracy là metric history trong epoch, không phải đánh giá lại checkpoint trên train.

| Run ID / ngày | Mô hình / tiền xử lý | Siêu tham số (LR, batch, epochs thực chạy/tối đa, seed) | Train accuracy tại checkpoint chọn | Validation accuracy tại checkpoint chọn | Test accuracy / loss | Ghi chú / đường dẫn |
| --- | --- | --- | --- | --- | --- | --- |
| 20261030_163159_141552 / 2026-10-30 | CNN 2 khối; MNIST /255 | LR 0,001; batch 128; epochs 11/15; seed 42 | 99,3778% | 98,9500% | 99,1600% / 0,026190 | 46,99 giây CPU; checkpoint epoch 8; `models/20261030_163159_141552/best.keras`; reload đạt |

Mỗi run cần ghi thêm trong `reports/experiments/<run_id>/`: optimizer, loss, cách chia dữ liệu, augmentation, dropout, epoch checkpoint tốt nhất, thời gian, thiết bị, phiên bản thư viện, history và đường dẫn mô hình. Ghi rõ accuracy dưới dạng phần trăm hoặc số trong `[0,1]`, không trộn hai cách.

Chỉ đo test cho mô hình đã chọn bằng validation; các run chưa đo test giữ `—`. Không chọn checkpoint hoặc điều chỉnh siêu tham số theo test.

### Kiểm tra demo và ảnh thực tế

| Chế độ | Trạng thái | Dữ liệu / điều kiện kiểm tra | Kết quả / bằng chứng |
| --- | --- | --- | --- |
| Ảnh PNG/JPG | Đã triển khai và chạy CLI | `data/samples/synthetic_1.png` do OpenCV tạo | Dự đoán 1; chưa đánh giá ảnh viết tay thật |
| Canvas | Đã kiểm tra tích hợp tự động | Tkinter thật, model thật; vẽ, xóa, dự đoán, ảnh trống | Đạt; chưa thử nghiệm người dùng |
| Webcam | Đã triển khai và kiểm tra giả lập | Camera index 0 không mở được; xem `reports/camera_check.json` | Luồng ROI/predict/thoát/thiếu thiết bị đạt; chưa xác minh webcam thật |

Lưu số mẫu, số dự đoán đúng và mẫu lỗi khi đánh giá ảnh thực tế. Không dùng accuracy MNIST thay cho kết quả demo.

## 5. Việc cần làm tiếp theo (Next steps)

1. Mở run_demo.cmd, mặc định Dãy số rời; xoay ảnh để các số đứng đúng hướng. Hai ảnh phát triển đã đạt chuỗi kỳ vọng; kiểm tra tiếp trên ảnh mới độc lập, đặc biệt chữ viết cách xa nhau giống hàng số.
2. Thu thập ảnh thật có box/nhãn chữ–số, tách theo người viết; đo bỏ sót và báo nhầm trước khi thay mô hình mặc định. EMNIST thuần/YOLO62 vẫn là phương pháp so sánh chưa đạt trên ảnh thực.
3. Bổ sung thông tin sinh viên, mẫu báo cáo/hạn nộp; quay demo và đóng gói mã, checkpoint, báo cáo, kết quả theo yêu cầu giảng viên.

**Điểm bắt đầu phiên tiếp theo:** runtime.json chọn numeric-rows (Dãy số rời). Checkpoint EMNIST tại emnist_candidate.json, YOLO62 tại yolo_emnist_candidate.json, baselineYOLO giữ nguyên. Không có tiến trình train chạy dở. Đã mở GUI bản mới trên desktop bằng ảnh gốc điện thoại và xác minh3512/4 box. Xem reports/FINAL_REPORT.md và reports/real_photo_regression.json trước khi tiếp tục.

## 6. Vấn đề / rủi ro đang mở

| Vấn đề / rủi ro | Tác động | Hướng xử lý / điều kiện đóng |
| --- | --- | --- |
| Đã đóng: môi trường | Python 3.11.9, TF 2.20.0, CPU | `pip check` đạt; `requirements-lock.txt` ghi phiên bản |
| Đã đóng: tải MNIST | Đã tải và chia đúng dữ liệu | Cache tại `data/raw/keras/datasets/mnist.npz`; có thống kê và hình minh họa |
| Ảnh thực tế khác phân bố MNIST | Accuracy test cao nhưng demo có thể sai | Kiểm tra cực tính, crop, padding và căn giữa; thu thập bộ ảnh thực tế có nhãn riêng |
| Webcam thật chưa khả dụng | Camera index 0 không mở được | Tkinter và luồng webcam giả lập đã đạt; cần kiểm tra thiết bị thật |
| Bài DataFlair chặn truy cập trực tiếp (HTTP 403) | Chưa đối chiếu được toàn bộ nội dung bài | Giữ liên kết gốc; đọc lại khi truy cập được, dùng tài liệu Keras cho thông tin kỹ thuật đã xác nhận |
| Chưa có hạn nộp, mẫu báo cáo và yêu cầu đóng gói của giảng viên | Chưa chốt được sản phẩm nộp cuối cùng | Bổ sung khi có thông tin; trước mắt viết báo cáo Markdown và chuẩn bị demo cục bộ |

## 7. Quy trình cập nhật hằng ngày

1. Đọc `PROGRESS.md` trước rồi `PROMPT.md`; đối chiếu với file và kết quả hiện có.
2. Tóm tắt trạng thái, chọn bước tiếp theo trong Next steps và thực hiện công việc.
3. Sau mỗi bước: cập nhật ngày sửa, đánh dấu checklist nếu đủ điều kiện, thêm nhật ký và số liệu thật nếu có.
4. Rút gọn Next steps thành các việc khả thi theo thứ tự; ghi rõ việc đang dở và cách tiếp tục.
5. Cập nhật hoặc đóng vấn đề mở khi có bằng chứng; không làm lại phần đã hoàn thành nếu không cần sửa lỗi.

Mẫu nhật ký cho lần cập nhật sau (chỉ điền khi đã thực hiện):

```text
| YYYY-MM-DD | Công việc thực tế | File/kết quả và cách xác minh | Lỗi còn mở hoặc Không có |
```

### 2026-11-01 — phân biệt chữ và số

Đã tạo 1.000 train/150 validation/100 test ảnh tổng hợp trộn MNIST và chữ Latin nhiều font; 20% lượt ảnh chỉ có chữ, 4% trống. Lớp10 other_letter có nhãn để học phân biệt; giao diện chỉ xuất lớp0–9. Đang fine-tune12 epochs; giữ checkpoint cũ, chỉ lưu candidate trước khi so sánh validation. Chưa có kết quả mới. Chữ tổng hợp chưa đại diện chữ viết tay thật; O/0, I/1 có thể không phân biệt được chỉ bằng hình dạng.


### Kết quả 2026-11-01

Đã fine-tune12 epochs và chọn mô hình mixed_20261101_130618 theo validation; đưa vào yolo_latest.json. Mô hình cũ lưu ở yolo_previous.json. Test tổng hợp100 ảnh,213 số và266 chữ: precision số41,86%→95,12%; recall95,31%→91,55%; chữ bị nhận thành số221→1 (83,08%→0,38%). Đây là so sánh cùng ngưỡng0.25, imgsz640, IoU matching0.5, tính đúng cả lớp. Kết quả chỉ áp dụng dữ liệu tổng hợp; chưa chứng minh độ chính xác trên ảnh giấy thật. Giảm báo nhầm đi kèm bỏ sót số nhiều hơn. Ảnh minh họa reports/figures/mixed_digits_demo.png. Việc tiếp theo: thu thập ảnh giấy trộn chữ viết tay và số, gán box cả hai loại, tách theo người viết để đánh giá/fine-tune. Không cần chạy lại baseline.


### Rollback theo phản hồi người dùng — 2026-11-01

Người dùng báo nhận số kém hơn trước sau khi đổi mixed. Đã khôi phục `models/yolo_latest.json` về `digits_20261031_180956` từ `yolo_previous.json`; giữ mixed tại `yolo_candidate.json`. Xác minh đường dẫn và file checkpoint, chạy lại demo đọc05221859 và3/3 kiểm thử YOLO đạt. Không khẳng định đã sửa được ảnh thực tế của người dùng. Việc chọn mixed mặc định từ kết quả tổng hợp là chưa đủ: recall test tổng hợp cũng đã giảm95,31%→91,55%. Chưa xác định nguyên nhân cụ thể trên ảnh thật do chưa có ảnh lỗi. Cửa sổ đang mở giữ mô hình trong RAM, cần đóng/mở lại để đổi checkpoint. Không tự áp dụng lại candidate dựa riêng trên chỉ số tổng hợp; cần kiểm tra hồi quy số và lỗi chữ→số bằng ảnh đại diện thực tế.


### Ảnh người dùng maxresdefault (1).jpg — 2026-11-01

Đã chạy ảnh gốc 1280×720 có giấy ô ly,9 số rõ(12345/6789), watermark đỏ và đầu bút. YOLO baseline xuất chuỗi sai2/2245/98928 (10 box, gồm watermark và bút); mixed chỉ xuất8. Đây là bằng chứng cả hai chưa tốt trên ảnh thực. Thử tách mực xanh/tím bằng HSV để loại nền xám và watermark đỏ; dùng connected components + CNN baseline nhận từng nét. Lọc mảnh thấp hơn0,5 trung vị chiều cao để bỏ nét chưa hoàn chỉnh cạnh bút. Kết quả chế độ blue-ink:9 box đúng nhãn12345/6789, không box watermark/bút; xem reports/figures/user_example_ink.png và JSON. Đây là ảnh phát triển đã dùng điều chỉnh thuật toán, không phải test độc lập; không huấn luyện lại hoặc thay con trỏ YOLO. Chế độ chỉ là tùy chọn checkbox trên GUI, không bảo đảm bỏ chữ cùng màu mực hoặc giữ số nhỏ khác cỡ. Hai kiểm thử bổ sung (ô ly + mực tím + chữ đỏ, chữ đỏ không có số) đạt. Việc tiếp theo: thử thêm ảnh thật chưa dùng chỉnh, gồm chữ cùng màu số và mực đen; cần dataset chữ viết tay/ảnh giấy thật thay cho font tổng hợp.


### Test ảnh điện thoại người dùng — 2026-11-01

Ảnh `1793514116442_2209075521745795385_8089266940901408211_d5688339a935afbe7d6fb5b0681671ad.jpg` có4 số3512 và chữ Xin chào cùng mực tím. Đã test3 chế độ trên hướng gốc và xoay90° ngược chiều kim đồng hồ (6 lượt). Hướng gốc: YOLO xuất5/5/5, mixed không tìm thấy, blue-ink xuất4. Hướng đứng: YOLO xuất2/332 (2 nhãn đúng ở3 và2, 5→3, thiếu1; box2 trên vân gỗ); mixed không tìm thấy; blue-ink xuất4/1, cả hai box nằm trên chữ, không trên4 số. Đã xem trực tiếp ảnh annotate để đối chiếu. JSON đầy đủ: reports/user_photo2_test.json; ảnh: reports/figures/user_photo2_upright_yolo.png và user_photo2_upright_blue_ink.png. Không chỉnh threshold, không huấn luyện hoặc thay checkpoint trong phép test này. Xoay đúng chiều chưa giải quyết lỗi; mực nhạt/cùng màu chữ, số nhỏ so với toàn cảnh và nền bàn là các khác biệt cần điều tra. Kết quả trước trên giấy ô ly không tổng quát. Cần giữ ảnh này làm regression case và đánh giá cả bỏ sót số lẫn nhận chữ/vật thể thành số. Các con số3512 là nhãn đối chiếu thủ công, không phải output đã sửa.


### Tiếp tục EMNIST — 2026-11-01

Đã tải EMNIST ByClass từ NIST qua torchvision (MD5 được thư viện kiểm tra), kiểm tra transpose/hướng ký tự bằng reports/figures/emnist_preview.png. Đang train CNN62 lớp, train/validation tách theo chỉ số từ train chính thức; test chỉ đo sau chọn checkpoint. Đồng thời chuẩn bị YOLO62 lớp trên ảnh ghép ký tự viết tay thật, có nét nhạt/giấy ô ly. Chưa thay mặc định. GUI thêm xoay hai hướng, lựa chọn mô hình và trạng thái không có số.


### Bản hoàn thiện mã và tài liệu — 2026-11-01

- Tải EMNIST chính thức, train classifier62 lớp15 epochs:152.692 train/12.400 validation; test116.323 ký tự. Accuracy62 lớp81,24%, digit correct-and-accepted71,31% và chữ→số0,81% ở ngưỡng0.65. Đây là ký tự tách sẵn; không phải accuracy ảnh giấy. Run emnist_20261101_133054 có history, confusion matrix, curves, split indices và checkpoint.
- Train YOLO62 từ ký tự EMNIST ghép:1.600/200/100 ảnh,12epochs. mAP50 test73,93%, mAP50–9572,70% trên toàn62 lớp. YOLO62 chưa tốt trên ảnh thật, không dùng mặc định.
- Bổ sung phân vùng màu thích ứng/nét nhạt và chế độ Dãy số rời:≥3 ký tự cùng cỡ/cách rộng, dùng CNN digit cho hàng số, EMNIST cho trường hợp không có hàng. Chính xác hai chuỗi phát triển123456789 và3512; chữ Xin chào/watermark/bàn không có box qua xem annotate. Đây là giả định bố cục; chữ viết cách xa vẫn có thể bị nhận thành số. Không tuyên bố loại chữ tổng quát hoặc100% accuracy.
- Đã so sánh4 phương pháp trên hai ảnh gốc, lưu reports/real_photo_regression.json và các ảnh final_*.png. Ảnh được dùng chỉnh heuristic nên không gọi là test độc lập. Ảnh JPEG dẫn xuất có nén lại từng nhận351; đã ghi giới hạn nhạy resize/nén trong báo cáo. GUI cuối dùng ảnh gốc, xoay trong RAM.
- 26 kiểm thử toàn bộ đạt; sau cập nhật GUI thêm1 kiểm thử trạng thái không có số/xoay trái đạt. pip check đạt; CLI help/compile/diff check đạt. Đã kiểm tra cửa sổ thật và kết quả3512.
- Hoàn thiện README, reports/FINAL_REPORT.md, script package_project. Đã đóng gói ZIP chứa mã, tài liệu, checkpoint và minh chứng; loại .venv/dataset thô. Kiểm tra integrity, file bắt buộc, con trỏ checkpoint và compile mã trong ZIP đều đạt. Chưa có video demo hoặc thông tin sinh viên/mẫu báo cáo/hạn nộp, chưa nộp bài. Công việc tiếp theo là kiểm tra ảnh độc lập và bổ sung yêu cầu môn học, không cần train lại những run đã lưu.


### Đơn giản hóa giao diện — 2026-11-01

Người dùng không muốn chọn phương pháp mỗi ảnh. Đã thu gọn toàn bộ phương pháp/bộ lọc vào mục Nâng cao; giao diện chính dùng sẵn cấu hình mặc định và chỉ cần Mở ảnh → xoay nếu cần → Nhận dạng. Đóng Nâng cao khôi phục mặc định/tắt bộ lọc. Bỏ tên thuật toán khỏi tiêu đề cửa sổ. Đã chạy lại5 kiểm thử liên quan GUI/ảnh và đều đạt; không thay trọng số hoặc thuật toán suy luận. Cần mở lại cửa sổ để thấy giao diện mới.


### Sửa phân vùng ảnh ghép ô sáng — 2026-11-01

Ảnh image-15.png kỳ vọng50419/21314. Trước sửa cấu hình mặc định chỉ xuất54 vì pixel nhiễu màu làm kích hoạt lọc màu và mép ô sáng trở thành thành phần nét. Đã thêm phát hiện tổng quát>=3 ô chữ nhật sáng, kích thước tương tự trên viền/nền tối; bỏ viền ô và threshold grayscale riêng từng ô. Không giả định số hàng/cột hoặc hardcode nhãn. Chế độ tự kích hoạt trong prepare_page trước lọc màu. Sau sửa xuất đúng50419/21314,10 box; xem reports/figures/image15_after.png và JSON. Hai ảnh cũ chạy hồi quy numeric-rows vẫn đúng123456789/3512. Sáu kiểm thử EMNIST/tiền xử lý đạt, gồm2 trường hợp mới về loại viền và không kích hoạt khi quá ít ô. Đã thêm ảnh này vào check_real_photos như ca phát triển thứ3. Không thay trọng số hoặc bắt người dùng chọn phương pháp. Cần mở lại cửa sổ để nạp mã mới; ảnh đã dùng sửa lỗi nên không gọi là test độc lập.


### Sửa bỏ sót mực đỏ trên OIP (1).webp — 2026-11-01

Đã tái hiện cấu hình mặc định trả0 box. Ảnh474×266 có bảng0–9, chữ đỏ và số1 trong tiêu đề. Mục tiêu hồi quy của ca này là10 số trong bảng; chưa yêu cầu/kiểm chứng số1 nằm trong tiêu đề. Bộ lọc trước chỉ chọn xanh/tím; nét số nhỏ còn có viền đen và nhiễu nén nên mask màu đơn thuần làm mất nét/lấp ruột ký tự. Bổ sung đề xuất màu đỏ độc lập, chọn dải màu theo diện tích các vùng nét không chạm biên thay vì gộp ngưỡng chung (bản gộp từng làm ảnh điện thoại sai371, đã loại bỏ). Khi chọn đỏ, dùng màu làm seed rồi giữ thành phần nét tối lân cận trên ảnh chuẩn hóa nền; threshold từ phân bố độ tối ở seed. Ảnh nhỏ không dùng closing3px để tránh lấp ruột ký tự. Không hardcode nhãn, không thay checkpoint.

Kết quả cuối:01234/56789,10 box, xem reports/figures/oip_after.png. Bốn ảnh phát triển hồi quy đều đúng chuỗi bảng/dãy kỳ vọng:0123456789,5041921314,123456789,3512. Bảy kiểm thử EMNIST/tiền xử lý đạt, có ca mực đỏ giữ viền đen. Đây là điều chỉnh trên ảnh phát triển đã biết; không phải test độc lập, chưa chứng minh hỗ trợ mọi bố cục hoặc ảnh mực nhiều màu. Tự xử lý trong cấu hình mặc định, không yêu cầu người dùng chọn phương pháp; mở lại ứng dụng để nạp mã mới.


### Ảnh dày OIP.webp — 2026-11-01

Ảnh474×265 có10 hàng0–9,20 ví dụ/hàng, tổng200 số theo đối chiếu bố cục thủ công. Bản trước chỉ xuất130 box, nhiều0/1/8 bị EMNIST loại như chữ, khoảng cách không đạt quy tắc dãy rời. Đã bổ sung nhận biết lưới dày:ít nhất3 hàng, mỗi hàng>=8 vùng, nhịp cột biến thiên tương đối<0.35 và khoảng cách tâm>0.7 chiều cao trung vị. Các nhãn vẫn do CNN dự đoán từng ô, không gán nhãn theo hàng/đa số. Gộp thành phần nét đứt nhỏ khi trang grayscale có>=50 thành phần, chồng lấp ngang/gap dọc đủ gần; giữ nguyên pixel foreground, không thêm nét hoặc gộp hàng kế bên. Nhãn vẽ nhỏ hơn theo chiều cao vùng.

Sau sửa tách đủ200 vùng, đọc đúng199/200 theo nhãn thủ công; lỗi còn lại hàng5 cột14:số4 đọc8, bbox313,111,14,18 và điểm0.9561. Giữ lỗi nguyên vẹn. Metrics reports/dense_evaluation.json; ảnh reports/figures/dense_after.png; crop gốc lỗi reports/figures/dense_error_original.png. 99,5% chỉ là trên ảnh phát triển này, không phải accuracy tổng quát; nguồn ảnh chưa rõ, có thể trùng dữ liệu MNIST. Bốn ảnh phát triển cũ vẫn đúng chuỗi kỳ vọng,8 kiểm thử liên quan đạt. Đã thêm ca dense vào check_real_photos với expected đầy đủ để test tiếp sẽ phản ánh exact_match=False. Lưới chữ cái cùng hình thức vẫn có thể bị nhận thành số; chưa giải quyết loại chữ tổng quát.


### Dọn dự án chuẩn bị nộp — 2026-11-01

Theo yêu cầu người dùng, kiểm kê đường dẫn/phụ thuộc trước khi xóa. Đã xóa143 file dư (~1945,16MiB): archive/split EMNIST không dùng, cache, log, last checkpoint không tham chiếu, batch preview, mask/threshold diagnostics không dẫn trong tài liệu và6 ZIP cũ. Giữ nguyên src/tests/notebooks/requirements, best checkpoints được các con trỏ dùng, YOLO initialization, dữ liệu train đang dùng, .venv và .git. 51 file được bảo vệ có SHA256 không đổi. EMNIST ByClass train/test vẫn load offline697.932/116.323 mẫu. 31 tests đạt; kiểm tra5 ảnh giữ kết quả cũ (4 exact, ảnh dense vẫn199/200 với lỗi4→8 đã biết). Chi tiết reports/cleanup_result.json và reports/CLEANUP.md. Bản ZIP nộp duy nhất đặt ở dist/DU_AN_NHAN_DANG_CHU_SO.zip, loại venv/dataset thô/cache; cần kiểm tra integrity và các model pointers trong ZIP sau tạo. Không thay thuật toán hoặc kết quả nhận dạng trong lần dọn này.
