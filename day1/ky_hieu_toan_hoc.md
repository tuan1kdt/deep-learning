# Giải thích mọi ký hiệu toán học trong bài Backpropagation

Tài liệu này giải nghĩa **từng ký hiệu** xuất hiện trong công thức backprop và file
[backprop_proof.html](backprop_proof.html), nối với code trong
[02_mlp_backprop.py](02_mlp_backprop.py). Đọc kèm khi gặp ký hiệu lạ.

Công thức trung tâm ta đang mổ xẻ:

$$\delta^l = \big((W^{l+1})^\top\, \delta^{l+1}\big) \odot g'^{\,l}(z^l)$$

---

## 1. Chỉ số trên & chỉ số dưới (superscript / subscript)

Đây là thứ gây rối nhất cho người mới — phân biệt rõ:

| Ký hiệu | Đọc là | Nghĩa |
|---|---|---|
| $x^l$ | "x mũ l" — **chỉ số trên** | Số thứ tự **lớp** (layer). KHÔNG phải lũy thừa. $z^2$ = pre-activation của **lớp 2**, không phải $z$ bình phương. |
| $x_j$ | "x chỉ số j" — **chỉ số dưới** | Số thứ tự **phần tử/neuron** trong vector. $z^l_j$ = pre-activation của **neuron thứ $j$** ở **lớp $l$**. |
| $W^l_{kj}$ | | Phần tử ở **hàng $k$, cột $j$** của ma trận $W^l$. |

> **Cảnh báo:** chỉ số trên ở đây là *nhãn lớp*, không phải số mũ. Chỉ khi viết rõ như
> $(z)^2$ hay $g'$ mới là phép toán. Khi nghi ngờ, cứ hiểu chỉ số trên = "thuộc lớp nào".

Quy ước lớp: $l = 1, 2, \dots, L$. Lớp $L$ là lớp cuối (output). Lớp $l+1$ là lớp **ngay sau**
lớp $l$ (gần output hơn); lớp $l-1$ là lớp **ngay trước** (gần input hơn).

---

## 2. Các đại lượng chính

| Ký hiệu | Tên | Nghĩa | Trong code |
|---|---|---|---|
| $x$ hoặc $a^0$ | Đầu vào | Vector dữ liệu đầu vào của mạng | `X` |
| $z^l$ | **Pre-activation** | Tổ hợp tuyến tính *trước* khi qua hàm kích hoạt: $z^l = W^l a^{l-1} + b^l$ | `Z1`, `Z2` |
| $a^l$ | **Activation** | Giá trị *sau* khi qua hàm kích hoạt: $a^l = g^l(z^l)$ | `A1`, `A2` |
| $W^l$ | **Ma trận trọng số** | Trọng số nối lớp $l-1$ sang lớp $l$. Kích thước $(n_l \times n_{l-1})$ | `W1`, `W2` |
| $b^l$ | **Bias** (độ lệch) | Vector cộng thêm, dịch chuyển kết quả lên/xuống | `b1`, `b2` |
| $\hat{y}$ | y mũ | Giá trị **dự đoán** của mạng (= $a^L$) | `y_pred`, `A2` |
| $y$ | | Giá trị **thật** (nhãn) | `y` |
| $L$ | **Loss** (hàm mất mát) | Một con số đo độ sai. Càng nhỏ càng tốt | `loss`, `bce_loss(...)` |
| $n_l$ | | Số neuron của lớp $l$ | `n_hidden`, ... |

**Quan hệ forward (lan truyền xuôi)** — đọc từ trái sang phải:

$$a^{l-1} \;\xrightarrow{\;\times W^l,\ +b^l\;}\; z^l \;\xrightarrow{\;g^l\;}\; a^l$$

---

## 3. $\delta$ (delta) — "sai số" tại một lớp

$$\delta^l \equiv \frac{\partial L}{\partial z^l}$$

| | |
|---|---|
| Đọc | "delta mũ l" |
| Nghĩa | **Độ nhạy của loss theo pre-activation $z^l$**: vặn $z^l$ lên một chút thì loss đổi bao nhiêu |
| KHÔNG phải | Không phải sai số kiểu $y - \hat y$. Nó là một **đạo hàm** |
| Trong code | `dZ1`, `dZ2` (biến `dZ...` chính là $\delta$ của lớp đó) |

$\delta^l_j$ (có chỉ số dưới) = thành phần thứ $j$ của vector đó = độ nhạy của loss
theo neuron $j$ ở lớp $l$.

---

## 4. $g$ và $g'$ — hàm kích hoạt và đạo hàm của nó

| Ký hiệu | Tên | Nghĩa |
|---|---|---|
| $g^l(\cdot)$ | **Hàm kích hoạt** (activation function) của lớp $l$ | Hàm phi tuyến áp lên từng phần tử, ví dụ ReLU hay sigmoid. Tạo "độ cong" cho mạng |
| $g'^{\,l}(z^l)$ | **Đạo hàm** của hàm kích hoạt | Dấu phẩy `′` nghĩa là "đạo hàm". $g'$ = "độ dốc" của $g$ — đo độ nhạy: tại điểm $z$ này, $g$ thay đổi nhanh hay chậm |

Hai hàm kích hoạt trong bài:

| Hàm | Công thức | Đạo hàm | Code |
|---|---|---|---|
| **ReLU** | $g(z) = \max(0, z)$ | $g'(z) = \begin{cases}1 & z>0\\ 0 & z\le 0\end{cases}$ | `relu`, `relu_grad` |
| **Sigmoid** | $\sigma(z) = \dfrac{1}{1+e^{-z}}$ | $\sigma'(z) = \sigma(z)\,(1-\sigma(z))$ | `sigmoid` |

> Dấu phẩy phẩy trong $g'^{\,l}$ rất dễ nhầm với chỉ số. Tách ra: $g'$ (đạo hàm) +
> $^l$ (của lớp $l$), áp lên $(z^l)$.

---

## 5. Các phép toán & dấu

### $\odot$ — Nhân từng phần tử (Hadamard product)

$$\begin{bmatrix} a \\ b \\ c \end{bmatrix} \odot \begin{bmatrix} x \\ y \\ z \end{bmatrix}
= \begin{bmatrix} a\cdot x \\ b\cdot y \\ c\cdot z \end{bmatrix}$$

Nhân **tương ứng vị trí**, KHÁC hẳn nhân ma trận. Hai vector phải cùng kích thước.
Trong code chính là phép `*` của NumPy: `dA1 * relu_grad(Z1)`.

### $(\;)^\top$ — Chuyển vị (transpose)

Lật ma trận qua đường chéo: hàng thành cột.

$$\begin{bmatrix} a & b \\ c & d \\ e & f \end{bmatrix}^\top
= \begin{bmatrix} a & c & e \\ b & d & f \end{bmatrix}$$

Kích thước $(3\times 2)$ thành $(2 \times 3)$. Trong backprop: forward dùng $W$ (đi tới),
backward dùng $W^\top$ (đi lui). Code: `W2.T`.

### Nhân ma trận (ký hiệu ghép liền $W a$, hoặc `@` trong code)

$$(m \times n) \cdot (n \times p) = (m \times p)$$

Quy tắc vàng: **hai số ở GIỮA phải bằng nhau** (chúng triệt tiêu), lấy hai số NGOÀI làm
kích thước kết quả. Slide viết kiểu cột $z = W a$; code viết kiểu hàng `X @ W` nên thứ tự
và chuyển vị đổi vế tương ứng.

### $\sum_k$ — Tổng (sigma)

$$\sum_{k} f(k) = f(1) + f(2) + f(3) + \dots$$

"Cộng dồn $f(k)$ khi $k$ chạy qua mọi giá trị". Chỉ số $k$ là **biến chạy** (dummy):
nó biến mất sau khi cộng xong. Trong code thường là phép `.sum(...)` hay ẩn trong `@`.

---

## 6. Ký hiệu giải tích (đạo hàm)

| Ký hiệu | Tên | Nghĩa |
|---|---|---|
| $\dfrac{\partial L}{\partial z}$ | **Đạo hàm riêng** (partial derivative) | "$L$ đổi bao nhiêu khi vặn riêng $z$, giữ mọi thứ khác cố định". Dùng $\partial$ (không phải $d$) khi hàm có nhiều biến |
| $\dfrac{dy}{dx}$ | Đạo hàm thường | Khi chỉ có một biến |
| $\nabla_a L$ | **Gradient** (nabla) | Vector gom tất cả đạo hàm riêng: $\big(\frac{\partial L}{\partial a_1}, \frac{\partial L}{\partial a_2}, \dots\big)$. "La bàn" chỉ hướng loss tăng nhanh nhất |
| $g'$ | Đạo hàm (dấu phẩy) | Cách viết gọn $\dfrac{dg}{dz}$ cho hàm một biến |

### Chain rule (quy tắc đạo hàm hàm hợp) — linh hồn của backprop

Khi $x \to u \to y$ (dây chuyền phụ thuộc):

$$\frac{\partial y}{\partial x} = \frac{\partial y}{\partial u}\cdot\frac{\partial u}{\partial x}$$

"Nhân các tỉ lệ thay đổi dọc theo dây chuyền". Khi $x$ ảnh hưởng $y$ qua **nhiều** đường
trung gian $u_k$ thì cộng các đường lại:

$$\frac{\partial y}{\partial x} = \sum_k \frac{\partial y}{\partial u_k}\cdot\frac{\partial u_k}{\partial x}$$

Đây chính là Bước 1 trong chứng minh — lý do công thức $\delta$ có dấu $\sum_k$.

---

## 7. Ký hiệu phụ trợ khác

| Ký hiệu | Nghĩa |
|---|---|
| $\equiv$ | "được định nghĩa là" (định nghĩa, không phải kết quả tính ra) |
| $\approx$ | "xấp xỉ bằng" (ví dụ trong sai phân hữu hạn) |
| $\blacksquare$ | "kết thúc chứng minh" (Q.E.D.) |
| $\varepsilon$ | epsilon — một số dương **rất nhỏ** (ví dụ $10^{-6}$ trong sai phân, hay $10^{-9}$ chống $\log(0)$) |
| $\boxed{\;}$ | Đóng khung — nhấn mạnh kết quả/định nghĩa quan trọng |
| $e^{-z}$ | Số Euler $e \approx 2.718$ mũ $-z$ (trong công thức sigmoid) |
| $\log$ | Logarit tự nhiên (trong BCE loss) |

---

## 8. Bảng tra nhanh: ký hiệu toán ↔ biến code

| Toán học | Code (`02_mlp_backprop.py`) | Ghi chú |
|---|---|---|
| $z^1$ | `Z1` | pre-activation lớp ẩn |
| $a^1$ | `A1` | sau ReLU |
| $z^2$ | `Z2` | pre-activation lớp ra |
| $a^2 = \hat y$ | `A2` | sau sigmoid = dự đoán |
| $\delta^2$ | `dZ2` | $= (A2 - y)/n$ (sigmoid+BCE rút gọn) |
| $\delta^1$ | `dZ1` | $=$ `dA1 * relu_grad(Z1)` |
| $(W^2)^\top \delta^2$ | `dZ2 @ W2.T` → `dA1` | lỗi kéo ngược qua $W$ |
| $g'^{\,1}(z^1)$ | `relu_grad(Z1)` | đạo hàm ReLU |
| $\odot$ | `*` (element-wise) | nhân từng phần tử |
| $\dfrac{\partial L}{\partial W^l}$ | `dW1`, `dW2` | gradient trọng số |
| $\dfrac{\partial L}{\partial b^l}$ | `db1`, `db2` | gradient bias |

> Mẹo nhớ: trong code, biến bắt đầu bằng **`d`** (như `dZ2`, `dW1`) luôn là **đạo hàm của loss
> theo đại lượng đó** — tức là $\dfrac{\partial L}{\partial(\cdot)}$.

---

## Tóm tắt một câu

Mọi ký hiệu quy về ba nhóm: **đại lượng** ($z, a, W, b, \delta$ với chỉ số trên = lớp,
chỉ số dưới = neuron), **phép toán** ($\odot$ nhân phần tử, $^\top$ chuyển vị, $\sum$ tổng,
nhân ma trận), và **giải tích** ($\partial$ đạo hàm riêng, $\nabla$ gradient, $g'$ đạo hàm,
chain rule). Nắm ba nhóm này thì đọc công thức backprop nào cũng trôi.
