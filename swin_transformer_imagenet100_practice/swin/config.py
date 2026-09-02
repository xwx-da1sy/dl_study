"""ImageNet-100 四阶段自定义 Swin 网络的集中配置。"""

from pathlib import Path


# 当前文件位于 swin_transformer_imagenet100_practice/swin/config.py，
# parent.parent 因此指向整个 Swin 实践项目目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# 模型训练后使用的目录先集中定义，后续文件不要各自拼接路径。
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "custom_swin_optimized_best.pt"

RESULTS_DIR = PROJECT_ROOT / "results"
TRAINING_HISTORY_PATH = RESULTS_DIR / "optimized_training_history.json"
EVALUATION_SUMMARY_PATH = RESULTS_DIR / "evaluation_summary.json"

# 从 ImageNet-100 的 train 中每类固定取50张作为内部验证集。
# 官方 val 不参与模型选择，只留给最终评估。
VALIDATION_SIZE = 5_000

BATCH_SIZE = 128
NUM_WORKERS = 4
RANDOM_SEED = 42

# ImageNet-100 输入配置。
IMAGE_SIZE = 224
IN_CHANNELS = 3
NUM_CLASSES = 100

# 自定义 Swin 的结构配置；网络层在 embedding.py、encoder.py 和 model.py 中创建。
# patch_size=4 会把 224 x 224 图片变成 56 x 56 个 patch tokens。
PATCH_SIZE = 4
EMBED_DIM = 96
WINDOW_SIZE = 7

# 四个 Stage 使用 Swin-T 的深度、注意力头数量和通道维度。
STAGE_DEPTHS = (2, 2, 6, 2)
STAGE_NUM_HEADS = (3, 6, 12, 24)
STAGE_DIMS = (96, 192, 384, 768)

# Stage 4 只有一个 7 x 7 窗口，移动窗口不会产生跨窗口通信。
STAGE_USE_SHIFTED_WINDOWS = (True, True, True, False)

MLP_RATIO = 4.0
# Mixup、CutMix 和 DropPath 已经提供了较强正则化，因此不再叠加普通 Dropout。
DROPOUT_RATE = 0.0
ATTENTION_DROPOUT_RATE = 0.0
STOCHASTIC_DEPTH_RATE = 0.1

# Random Erasing 保留较轻强度，避免和 RandAugment、Mixup/CutMix 叠加过强。
RANDOM_ERASING_PROBABILITY = 0.1

# 训练参数集中存放；优化器在 optimization.py 中创建。
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
NUM_EPOCHS = 300
WARMUP_EPOCHS = 15
MIN_LEARNING_RATE = 1e-6
LABEL_SMOOTHING = 0.05

# 网格搜索先用较短训练比较组合，确定参数后再完整训练300轮。
SEARCH_EPOCHS = 100
LEARNING_RATE_CANDIDATES = (2e-4, 3e-4, 5e-4)
WEIGHT_DECAY_CANDIDATES = (0.02, 0.05)

# Mixup 混合整张图片，CutMix 只交换矩形区域。
# 两者都开启时，每个训练 batch 以50%的概率选择 CutMix，否则选择 Mixup。
MIXUP_ALPHA = 0.1
CUTMIX_ALPHA = 1.0
CUTMIX_PROBABILITY = 0.5

MAX_GRAD_NORM = 1.0
LOG_INTERVAL = 50
