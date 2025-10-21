# micrograd

![awww](puppy.jpg)

Một công cụ Autograd nhỏ bé (có một cú cắn! :)). Triển khai lan truyền ngược (tự động phân biệt chế độ ngược) trên một DAG được xây dựng động và một thư viện mạng thần kinh nhỏ gọn bên trên nó với API giống như PyTorch. Cả hai đều rất nhỏ gọn, chỉ khoảng 100 và 50 dòng mã tương ứng. DAG chỉ hoạt động với các giá trị vô hướng, ví dụ: chúng ta chia mỗi nơ-ron thành tất cả các phép cộng và nhân nhỏ lẻ của nó. Tuy nhiên, điều này đủ để xây dựng toàn bộ mạng thần kinh sâu thực hiện phân loại nhị phân, như sổ tay demo đã chỉ ra. Có thể hữu ích cho mục đích giáo dục.

### Cài đặt

```bash
pip install micrograd
```

### Ví dụ sử dụng

Dưới đây là một ví dụ hơi phức tạp cho thấy một số hoạt động được hỗ trợ có thể:

```python
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # in ra 24.7041, kết quả của lượt chuyển tiếp này
g.backward()
print(f'{a.grad:.4f}') # in ra 138.8338, tức là giá trị số của dg/da
print(f'{b.grad:.4f}') # in ra 645.5773, tức là giá trị số của dg/db
```

### Huấn luyện một mạng thần kinh

Sổ tay `demo.ipynb` cung cấp một bản demo đầy đủ về việc huấn luyện bộ phân loại nhị phân mạng thần kinh 2 lớp (MLP). Điều này đạt được bằng cách khởi tạo một mạng thần kinh từ mô-đun `micrograd.nn`, triển khai một hàm mất mát phân loại nhị phân "max-margin" svm đơn giản và sử dụng SGD để tối ưu hóa. Như được trình bày trong sổ tay, sử dụng mạng thần kinh 2 lớp với hai lớp ẩn 16 nút, chúng ta đạt được ranh giới quyết định sau trên tập dữ liệu moon:

![2d neuron](moon_mlp.png)

### Truy vết / trực quan hóa

Để thuận tiện hơn, sổ tay `trace_graph.ipynb` tạo ra các hình ảnh trực quan graphviz. Ví dụ, hình dưới đây là một nơ-ron 2D đơn giản, đạt được bằng cách gọi `draw_dot` trên đoạn mã bên dưới, và nó hiển thị cả dữ liệu (số bên trái trong mỗi nút) và gradient (số bên phải trong mỗi nút).

```python
from micrograd import nn
n = nn.Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)
dot = draw_dot(y)
```

![2d neuron](gout.svg)

### Chạy kiểm thử

Để chạy các kiểm thử đơn vị, bạn sẽ phải cài đặt [PyTorch](https://pytorch.org/), mà các kiểm thử sử dụng làm tham chiếu để xác minh tính đúng đắn của các gradient đã tính toán. Sau đó chỉ cần:

```bash
python -m pytest
```

### Giấy phép

MIT