"""Tiny ViT 的集中配置。"""

from pathlib import Path


# 项目目录指向 vision_transformer_practice，数据集继续保存在原来的 data 目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# 训练时只保存验证集 loss 最低的模型，避免后期过拟合模型覆盖最佳 checkpoint。
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
EMBED_DIM = 256

# 注意力子层配置：256 / 4 = 64，因此每个注意力头处理 64 维特征。
NUM_HEADS = 4

# 配合较温和的 Mixup，使用适中的 Dropout，避免正则化过强导致欠拟合。
DROPOUT_RATE = 0.1

# MLP 隐藏层通常扩展为 embed_dim 的 4 倍：256 x 4 = 1024。
MLP_HIDDEN_DIM = EMBED_DIM * 4

# 增加 Encoder 深度，提升 CIFAR-10 特征建模能力。
NUM_ENCODER_BLOCKS = 6

# CIFAR-10 共有 10 个互斥类别，因此分类头输出 10 个类别分数。
NUM_CLASSES = 10

# 训练配置：AdamW 初始学习率、权重衰减和训练轮数。
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 0.05

# 使用较温和的 Mixup，避免训练准确率和有效监督信号被过度削弱。
MIXUP_ALPHA = 0.1

# 前几个 epoch 线性升高学习率，避免 Transformer 在训练初期不稳定。
WARMUP_EPOCHS = 10

# 给增大后的模型更充分的收敛时间。
NUM_EPOCHS = 300

# Label Smoothing 降低模型对训练标签的过度自信。
LABEL_SMOOTHING = 0.05

# Early Stopping：验证 loss 连续 20 个 epoch 没有实质改善时停止训练。
EARLY_STOPPING_PATIENCE = 50
EARLY_STOPPING_MIN_DELTA = 1e-5

# 余弦退火最终将学习率从 3e-4 平滑降低到 1e-6。
MIN_LEARNING_RATE = 1e-6

# 梯度裁剪可以避免训练初期偶发的梯度过大；每 50 个 batch 打印一次进度。
MAX_GRAD_NORM = 1.0
LOG_INTERVAL = 50
