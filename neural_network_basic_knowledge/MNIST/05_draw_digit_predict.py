from pathlib import Path
import importlib.util
import tkinter as tk
from tkinter import messagebox

import torch
from PIL import Image, ImageDraw
from torchvision import transforms


# 第五步：让用户自己写数字，然后让模型预测。
# 这个脚本会打开一个小窗口：
# 1. 用户用鼠标在黑色画布上写一个白色数字。
# 2. 点击“预测”按钮。
# 3. 程序把用户写的数字处理成接近 MNIST 的 28x28 图片。
# 4. 加载训练好的 MLP 模型，输出预测结果。


project_dir = Path(__file__).resolve().parent
model_path = project_dir / "mlp_mnist.pth"


# 选择运行设备。
# 预测单张图片时 CPU 也很快，有 GPU 就用 GPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 复用 02_define_mlp_model.py 里的 MLP 类。
# 因为文件名以数字开头，所以继续用 importlib 按文件路径导入。
model_module_path = project_dir / "02_define_mlp_model.py"
spec = importlib.util.spec_from_file_location("define_mlp_model", model_module_path)
mlp_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlp_model)

MLP = mlp_model.MLP


# 创建模型，并加载训练好的参数。
model = MLP().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# 这里的预处理要和训练时保持一致。
# 训练时用的是 ToTensor + Normalize((0.1307,), (0.3081,))。
# 用户画出来的图片最后也必须经过同样的处理，模型才能正确理解。
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


def resize_like_mnist(image):
    # 用户看到的是 280x280 的大画布。
    # 神经网络需要的是 28x28 的输入。
    #
    # 这里的核心思想是：
    # 1. 先找出用户真正写字的区域，也就是非黑色像素所在的格子。
    # 2. 把这个区域裁剪出来。
    # 3. 补成正方形，避免数字被拉伸变形。
    # 4. 按比例缩放到接近 MNIST 的大小。
    # 5. 放进 28x28 的中心，交给神经网络。

    # getbbox 会找到非黑色像素的边界框。
    # 如果用户什么都没写，bbox 会是 None。
    bbox = image.getbbox()
    if bbox is None:
        return None

    digit = image.crop(bbox)
    width, height = digit.size

    # 把裁剪出来的数字补成正方形，避免缩放时被压扁。
    side = max(width, height)
    square = Image.new("L", (side, side), color=0)
    left = (side - width) // 2
    top = (side - height) // 2
    square.paste(digit, (left, top))

    # MNIST 的数字通常不是撑满 28x28，而是周围有一些黑色边距。
    # 所以先把数字主体缩放到 20x20，再放进 28x28 的中心。
    digit_20 = square.resize((20, 20), Image.Resampling.LANCZOS)

    mnist_like = Image.new("L", (28, 28), color=0)
    mnist_like.paste(digit_20, (4, 4))

    return mnist_like


class DigitDrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST 手写数字预测")

        # 用户输入界面保持 280x280。
        # 这样鼠标写数字比较舒服。
        # 底层预测时再把笔迹映射成神经网络需要的 28x28。
        self.canvas_size = 280

        # 笔刷太细时，缩放到 28x28 后可能断裂。
        # 稍微粗一点更接近 MNIST 的笔画粗细。
        self.brush_size = 22

        # tkinter 的 Canvas 负责显示给用户看。
        self.canvas = tk.Canvas(
            root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="black",
            cursor="cross",
        )
        self.canvas.pack(padx=12, pady=12)

        # PIL Image 负责在内存里保存同样的绘制内容。
        # 预测时不从屏幕截图，而是直接使用这个 image，更稳定。
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.last_x = None
        self.last_y = None

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        button_frame = tk.Frame(root)
        button_frame.pack(pady=(0, 12))

        self.predict_button = tk.Button(button_frame, text="预测", command=self.predict)
        self.predict_button.pack(side=tk.LEFT, padx=6)

        self.clear_button = tk.Button(button_frame, text="清空", command=self.clear)
        self.clear_button.pack(side=tk.LEFT, padx=6)

        self.result_label = tk.Label(root, text="请在画布上写一个 0~9 的数字")
        self.result_label.pack(pady=(0, 12))

    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def draw_line(self, event):
        if self.last_x is None or self.last_y is None:
            return

        # 在 tkinter 画布上画线，用户能看到。
        self.canvas.create_line(
            self.last_x,
            self.last_y,
            event.x,
            event.y,
            fill="white",
            width=self.brush_size,
            capstyle=tk.ROUND,
            smooth=True,
        )

        # 在 PIL 图片上画同样的线，后面预测用它。
        self.draw.line(
            [self.last_x, self.last_y, event.x, event.y],
            fill=255,
            width=self.brush_size,
        )

        self.last_x = event.x
        self.last_y = event.y

    def stop_draw(self, event):
        self.last_x = None
        self.last_y = None

    def clear(self):
        # 清空界面画布。
        self.canvas.delete("all")

        # 清空内存里的图片。
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.result_label.config(text="请在画布上写一个 0~9 的数字")

    def predict(self):
        if not model_path.exists():
            messagebox.showerror("错误", f"没有找到模型文件：{model_path}")
            return

        mnist_like_image = resize_like_mnist(self.image)
        if mnist_like_image is None:
            messagebox.showwarning("提示", "请先写一个数字")
            return

        # transform(mnist_like_image) 得到形状 [1, 28, 28] 的 Tensor。
        # unsqueeze(0) 增加 batch 维度，变成 [1, 1, 28, 28]。
        input_tensor = transform(mnist_like_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prediction = output.argmax(dim=1).item()

            # softmax 可以把 logits 转成概率，方便用户理解模型有多确定。
            probabilities = torch.softmax(output, dim=1)
            confidence = probabilities[0, prediction].item()

        self.result_label.config(
            text=f"预测结果：{prediction}，置信度：{confidence * 100:.2f}%"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitDrawApp(root)
    root.mainloop()
