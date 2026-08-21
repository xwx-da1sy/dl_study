"""CNN 公平对照实验的独立输出路径。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CNN 与 TinyViT 使用不同输出文件，避免互相覆盖 checkpoint 和评估结果。
CNN_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "cnn_baseline_best.pt"
CNN_RESULTS_DIR = PROJECT_ROOT / "results" / "cnn_baseline"
CNN_HISTORY_PATH = CNN_RESULTS_DIR / "training_history.json"
CNN_EVALUATION_SUMMARY_PATH = CNN_RESULTS_DIR / "evaluation_summary.json"
CNN_TRAINING_CURVES_PATH = CNN_RESULTS_DIR / "training_curves.png"
CNN_CONFUSION_MATRIX_PATH = CNN_RESULTS_DIR / "confusion_matrix.png"
CNN_SAMPLE_PREDICTIONS_PATH = CNN_RESULTS_DIR / "sample_predictions.png"
