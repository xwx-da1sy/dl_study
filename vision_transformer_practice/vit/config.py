"""Tiny ViT 的集中配置。"""

from pathlib import Path


# 项目目录指向 vision_transformer_practice，数据集继续保存在原来的 data 目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# 训练时只保存验证集准确率最高的模型，避免每个 epoch 都生成大文件。
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "tiny_vit_best.pt"

# 训练历史、测试指标和可视化图片统一保存在 results 目录。
RESULTS_DIR = PROJECT_ROOT / "results"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.json"
EVALUATION_SUMMARY_PATH = RESULTS_DIR / "evaluation_summary.json"
TRAINING_CURVES_PATH = RESULTS_DIR / "training_curves.png"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
SAMPLE_PREDICTIONS_PATH = RESULTS_DIR / "sample_predictions.png"

# CIFAR-10 训练集共 50000 张图片，这里划分 45000 张训练、5000 张验证。
VALIDATION_SIZE = 5_000

# Tiny ViT 输入较小，先使用 batch_size=128；显存不足时可以再减小。
BATCH_SIZE = 128

# Windows 环境先使用单进程加载，代码最稳定；后续训练较慢时再增加。
NUM_WORKERS = 0

# 固定随机种子，保证每次运行得到相同的训练集和验证集划分。
RANDOM_SEED = 42

# Tiny ViT 输入与 Patch Embedding 配置。
IMAGE_SIZE = 32
PATCH_SIZE = 4
IN_CHANNELS = 3
EMBED_DIM = 192

# 注意力子层配置：192 / 3 = 64，因此每个注意力头处理 64 维特征。
NUM_HEADS = 3

# Dropout 概率为 0.1：训练时随机丢弃 10% 的元素，验证和测试时自动关闭。
DROPOUT_RATE = 0.1

# MLP 隐藏层通常扩展为 embed_dim 的 4 倍：192 x 4 = 768。
MLP_HIDDEN_DIM = EMBED_DIM * 4

# 当前 Tiny ViT 堆叠 4 个 Encoder Block；后续可以通过这个常量调整深度。
NUM_ENCODER_BLOCKS = 4

# CIFAR-10 共有 10 个互斥类别，因此分类头输出 10 个类别分数。
NUM_CLASSES = 10

# 训练配置：AdamW 初始学习率、权重衰减和训练轮数。
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
NUM_EPOCHS = 500

# 余弦退火最终将学习率从 3e-4 平滑降低到 1e-6。
MIN_LEARNING_RATE = 1e-6

# 梯度裁剪可以避免训练初期偶发的梯度过大；每 50 个 batch 打印一次进度。
MAX_GRAD_NORM = 1.0
LOG_INTERVAL = 50
