from pathlib import Path
import importlib.util

import torch


# 第四步：加载训练好的模型，并用测试集图片做预测。
# 前面训练脚本已经把模型参数保存到了 mlp_mnist.pth。
# 这个脚本的目标是：
# 1. 复用测试集 test_loader。
# 2. 复用 MLP 模型结构。
# 3. 加载 mlp_mnist.pth 里的模型参数。
# 4. 取几张测试图片，看看模型预测是否正确。


# 当前文件所在目录，也就是 MNIST 项目目录。
project_dir = Path(__file__).resolve().parent


# 模型参数保存路径。
# 这个文件应该由 03_train_mnist.py 训练结束后生成。
model_path = project_dir / "mlp_mnist_deeper.pth"


# 选择运行设备。
# 加载模型和预测时，也要保证模型、图片都在同一个 device 上。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 复用 01_load_mnist_data.py 里的测试集 DataLoader。
# 因为文件名以数字开头，所以仍然使用 importlib 按文件路径导入。
data_module_path = project_dir / "01_load_mnist_data.py"
spec = importlib.util.spec_from_file_location("load_mnist_data", data_module_path)
mnist_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mnist_data)

test_loader = mnist_data.test_loader


# 复用 02_define_mlp_model.py 里的 MLP 模型结构。
model_module_path = project_dir / "02_define_mlp_model.py"
spec = importlib.util.spec_from_file_location("define_mlp_model", model_module_path)
mlp_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlp_model)

MLP = mlp_model.MLP


# 创建一个新的 MLP 模型对象。
# 注意：这里只是创建了模型结构，里面的参数还是随机初始化的。
model = MLP().to(device)


# load_state_dict 会把训练好的参数加载进模型。
# torch.load(model_path, map_location=device) 表示从 mlp_mnist_deeper.pth 读取参数，
# 并把参数放到当前使用的 device 上。
model.load_state_dict(torch.load(model_path, map_location=device))


# 预测时要切换到评估模式。
# 评估模式下，模型不会执行训练阶段特有的行为。
model.eval()


if __name__ == "__main__":
    # 从测试集中取出一个 batch。
    # images 的形状是 [batch_size, 1, 28, 28]。
    # labels 是这些图片对应的真实数字标签。
    images, labels = next(iter(test_loader))

    # 模型在 device 上，所以图片也要移动到同一个 device。
    images = images.to(device)
    labels = labels.to(device)

    # 预测阶段不需要计算梯度。
    # torch.no_grad() 可以节省显存和计算时间。
    with torch.no_grad():
        # outputs 的形状是 [batch_size, 10]。
        # 每一行代表一张图片属于 0~9 的原始分数。
        outputs = model(images)

        # argmax(dim=1) 表示取每张图片分数最高的类别作为预测结果。
        predictions = outputs.argmax(dim=1)

    # 先打印前 10 张图片的真实标签和预测标签。
    # 如果模型训练得不错，这两行大部分位置应该相同。
    print("真实标签:", labels[:10].tolist())
    print("预测标签:", predictions[:10].tolist())
