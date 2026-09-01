"""只使用验证集搜索 AdamW 的学习率和 Weight Decay。"""

import argparse
import gc
import json
from itertools import product
from pathlib import Path

import torch

from swin import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    LABEL_SMOOTHING,
    LEARNING_RATE_CANDIDATES,
    LOG_INTERVAL,
    MAX_GRAD_NORM,
    MIN_LEARNING_RATE,
    NUM_WORKERS,
    RESULTS_DIR,
    SEARCH_EPOCHS,
    WARMUP_EPOCHS,
    WEIGHT_DECAY_CANDIDATES,
    CustomSwin,
    create_cifar100_dataloaders,
    create_loss_function,
    create_optimizer,
    create_warmup_cosine_scheduler,
    fit,
    set_random_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="在固定验证集上搜索 Swin 的学习率和 Weight Decay"
    )
    parser.add_argument("--epochs", type=int, default=SEARCH_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=LEARNING_RATE_CANDIDATES,
    )
    parser.add_argument(
        "--weight-decays",
        type=float,
        nargs="+",
        default=WEIGHT_DECAY_CANDIDATES,
    )
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="关闭 CUDA 自动混合精度",
    )
    return parser.parse_args()


def validate_args(args):
    if args.epochs <= 0:
        raise ValueError("epochs 必须大于0")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("warmup_epochs 必须位于[0, epochs)之间")
    if not args.learning_rates or any(value <= 0 for value in args.learning_rates):
        raise ValueError("所有 learning rate 都必须大于0")
    if not args.weight_decays or any(value < 0 for value in args.weight_decays):
        raise ValueError("所有 weight decay 都不能小于0")


def format_run_name(learning_rate, weight_decay):
    learning_rate_text = f"{learning_rate:.0e}".replace("-", "m")
    weight_decay_text = f"{weight_decay:g}".replace(".", "p")
    return f"lr_{learning_rate_text}_wd_{weight_decay_text}"


def save_summary(summary_path, args, trials):
    ranking = sorted(
        trials,
        key=lambda item: (
            -item["best_validation_accuracy"],
            item["best_validation_loss"],
        ),
    )
    summary = {
        "search_config": {
            "epochs_per_trial": args.epochs,
            "warmup_epochs": args.warmup_epochs,
            "learning_rates": list(args.learning_rates),
            "weight_decays": list(args.weight_decays),
            "label_smoothing": LABEL_SMOOTHING,
        },
        "best_trial": ranking[0] if ranking else None,
        "ranking": ranking,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
    return summary


def main():
    args = parse_args()
    validate_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    search_result_dir = RESULTS_DIR / "hyperparameter_search"
    search_checkpoint_dir = CHECKPOINT_DIR / "hyperparameter_search"
    summary_path = search_result_dir / "search_summary.json"
    combinations = list(product(args.learning_rates, args.weight_decays))
    trials = []

    print(f"共{len(combinations)}组参数；每组训练{args.epochs}轮")
    print("只比较固定验证集，官方测试集不会参与搜索")

    for trial_index, (learning_rate, weight_decay) in enumerate(
        combinations,
        start=1,
    ):
        # 每组都重置随机种子和 DataLoader，使比较尽量只受超参数影响。
        set_random_seed()
        train_loader, validation_loader, test_loader, _ = (
            create_cifar100_dataloaders(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        )
        run_name = format_run_name(learning_rate, weight_decay)
        checkpoint_path = search_checkpoint_dir / f"{run_name}.pt"
        history_path = search_result_dir / f"{run_name}_history.json"

        print("=" * 70)
        print(
            f"搜索 {trial_index}/{len(combinations)}："
            f"learning_rate={learning_rate:g}, weight_decay={weight_decay:g}"
        )

        model = CustomSwin().to(device)
        criterion = create_loss_function(LABEL_SMOOTHING)
        optimizer = create_optimizer(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = create_warmup_cosine_scheduler(
            optimizer=optimizer,
            num_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            min_learning_rate=MIN_LEARNING_RATE,
        )
        result = fit(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=args.epochs,
            checkpoint_path=checkpoint_path,
            history_path=history_path,
            use_amp=not args.no_amp,
            log_interval=args.log_interval,
            max_grad_norm=MAX_GRAD_NORM,
        )

        trial = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "best_epoch": result["best_epoch"],
            "best_validation_accuracy": result["best_validation_accuracy"],
            "best_validation_loss": result["best_validation_loss"],
            "checkpoint_path": str(checkpoint_path),
            "history_path": str(history_path),
        }
        trials.append(trial)
        save_summary(summary_path, args, trials)

        # 一组实验结束后及时释放显存，再创建下一组全新模型。
        del model, criterion, optimizer, scheduler
        del train_loader, validation_loader, test_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = save_summary(summary_path, args, trials)
    best = summary["best_trial"]
    print("=" * 70)
    print("网格搜索完成")
    print(
        f"最佳组合：learning_rate={best['learning_rate']:g}, "
        f"weight_decay={best['weight_decay']:g}"
    )
    print(f"最佳验证准确率：{best['best_validation_accuracy'] * 100:.2f}%")
    print(f"完整排名：{summary_path}")
    print("使用最佳组合运行300轮正式训练：")
    print(
        "python train_custom_swin.py "
        f"--learning-rate {best['learning_rate']:g} "
        f"--weight-decay {best['weight_decay']:g}"
    )


if __name__ == "__main__":
    main()
