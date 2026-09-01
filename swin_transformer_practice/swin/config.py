"""CIFAR-100 自定义 Swin 网络的集中配置。"""

from pathlib import Path


# 当前文件位于 swin_transformer_practice/swin/config.py，
# parent.parent 因此指向整个 Swin 实践项目目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# 模型训练后使用的目录先集中定义，后续文件不要各自拼接路径。
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "custom_swin_best.pt"

RESULTS_DIR = PROJECT_ROOT / "results"
TRAINING_HISTORY_PATH = RESULTS_DIR / "training_history.json"
EVALUATION_SUMMARY_PATH = RESULTS_DIR / "evaluation_summary.json"

# CIFAR-100 官方训练集有50,000张图片。
# 每个类别固定取50张验证，因此训练/验证数量为45,000/5,000。
VALIDATION_SIZE = 5_000

BATCH_SIZE = 128
NUM_WORKERS = 4
RANDOM_SEED = 42

# CIFAR-100 输入配置。
IMAGE_SIZE = 32
IN_CHANNELS = 3
NUM_CLASSES = 100

# 自定义 Swin 的结构配置；网络层在 embedding.py、encoder.py 和 model.py 中创建。
# patch_size=2 会把 32 x 32 图片变成 16 x 16 个 patch tokens。
PATCH_SIZE = 2
EMBED_DIM = 96
WINDOW_SIZE = 4

# 三个 Stage 的 Block 数量、注意力头数量和通道维度一一对应。
STAGE_DEPTHS = (2, 2, 2)
STAGE_NUM_HEADS = (3, 6, 12)
STAGE_DIMS = (96, 192, 384)

# 前两个 Stage 需要 SW-MSA 跨窗口通信；Stage 3 只有一个 4 x 4 窗口，无需移动。
STAGE_USE_SHIFTED_WINDOWS = (True, True, False)

MLP_RATIO = 4.0
DROPOUT_RATE = 0.1
ATTENTION_DROPOUT_RATE = 0.0
STOCHASTIC_DEPTH_RATE = 0.1

# 训练参数集中存放；优化器在 optimization.py 中创建。
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
NUM_EPOCHS = 200
WARMUP_EPOCHS = 10
MIN_LEARNING_RATE = 1e-6
LABEL_SMOOTHING = 0.1

# Mixup 混合整张图片，CutMix 只交换矩形区域。
# 两者都开启时，每个训练 batch 以50%的概率选择 CutMix，否则选择 Mixup。
MIXUP_ALPHA = 0.1
CUTMIX_ALPHA = 1.0
CUTMIX_PROBABILITY = 0.5

MAX_GRAD_NORM = 1.0
LOG_INTERVAL = 50
