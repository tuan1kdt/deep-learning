# Day 1 — Nền tảng Mạng Neural

## 1. Khái niệm cốt lõi & cấu trúc mạng (L tầng ẩn)

- **Bản chất:** Mạng neural mô phỏng cách bộ não con người học tập (thử $\rightarrow$ sai $\rightarrow$ sửa lỗi $\rightarrow$ rút ra quy luật).
- **Thành phần cơ bản:**
  1. **Trọng số ($W$):** thể hiện mức độ quan trọng của từng thông tin đầu vào.
  2. **Định thiên ($Bias$):**
     - Để biểu diễn được dữ liệu mà không đi qua gốc tọa độ.
     - Nếu không có mà giá trị đầu vào là 0 thì $W$ (trọng số) lớn bao nhiêu kết quả cũng bằng 0.
  3. **Hàm kích hoạt ($Activation\ Function$):**
     - Để biểu diễn được dữ liệu dạng cong (non-linear).
     - Sàng lọc dữ liệu: quyết định dữ liệu nào đáng giá để truyền tiếp vào tầng tiếp theo.

## 2. Phân biệt 2 bài toán

- **Bài toán Hồi quy (Regression):** Đầu ra là **1 nút duy nhất**, tính tổng tuyến tính $\Sigma$ để trả về một con số liên tục.
- **Bài toán Phân loại đa lớp (Multi-class Classification):** Đầu ra cần **$K$ nút** tương ứng với $K$ nhãn cần phân loại. Tầng này bắt buộc phải đi qua hàm kích hoạt **Softmax** để đưa ra dự báo dạng xác suất.

## 3. Hàm Loss

- Mục tiêu của quá trình huấn luyện là tối ưu hóa để Loss tiến về mức thấp nhất có thể.
- **Bài toán Hồi quy (Regression):** MSE — *Mean Squared Error*.
- **Bài toán Phân loại (Classification):** Cross-Entropy.

## 4. Thuật toán GDA (Gradient Descent Algorithm)

-

## 5. Bài tập về nhà
Tìm hiểu thuật toán Lan truyền ngược (backpropagation) trong huấn luyện mạng MLP (MultiLayer Perceptron), trình bày lý thuyết (chi tiết) và code