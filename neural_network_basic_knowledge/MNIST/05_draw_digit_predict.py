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
# 3. 程序把用户写的数字缩放成 28x28。
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
# 用户画出来的图片也必须经过同样的处理，模型才能正确理解。
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


class DigitDrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST 手写数字预测")

        # 画布大小设置成 280x280。
        # 因为 MNIST 是 28x28，这里相当于放大 10 倍，用户更容易写。
        self.canvas_size = 280
        self.brush_size = 18

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

        # 把 280x280 的手写图缩放到 MNIST 的 28x28。
        small_image = self.image.resize((28, 28), Image.Resampling.LANCZOS)

        # transform(small_image) 得到形状 [1, 28, 28] 的 Tensor。
        # unsqueeze(0) 增加 batch 维度，变成 [1, 1, 28, 28]。
        input_tensor = transform(small_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prediction = output.argmax(dim=1).item()

            # softmax 可以把 logits 转成概率，方便理解模型有多确定。
            probabilities = torch.softmax(output, dim=1)
            confidence = probabilities[0, prediction].item()

        self.result_label.config(
            text=f"预测结果：{prediction}，置信度：{confidence * 100:.2f}%"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitDrawApp(root)
    root.mainloop()
