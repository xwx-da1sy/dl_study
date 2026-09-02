"""自定义 Swin 的损失函数、学习率调度器、训练和验证流程。"""

import json
import math
from pathlib import Path

import torch
from torch import nn

from .config import (
    BEST_MODEL_PATH,
    CUTMIX_ALPHA,
    CUTMIX_PROBABILITY,
    LABEL_SMOOTHING,
    LOG_INTERVAL,
    MAX_GRAD_NORM,
    MIN_LEARNING_RATE,
    MIXUP_ALPHA,
    NUM_EPOCHS,
    RANDOM_SEED,
    TRAINING_HISTORY_PATH,
    WARMUP_EPOCHS,
)
from .data import apply_mixup_or_cutmix, calculate_mixed_loss


def create_loss_function(label_smoothing=LABEL_SMOOTHING):
    """创建带 Label Smoothing 的交叉熵损失。"""
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing 必须位于[0, 1)之间")
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def create_warmup_cosine_scheduler(
    optimizer,
    num_epochs=NUM_EPOCHS,
    warmup_epochs=WARMUP_EPOCHS,
    min_learning_rate=MIN_LEARNING_RATE,
):
    """先线性 Warmup，再用余弦曲线把学习率降到指定最小值。"""
    if num_epochs <= 0:
        raise ValueError("num_epochs 必须大于0")
    if warmup_epochs < 0 or warmup_epochs >= num_epochs:
        raise ValueError("warmup_epochs 必须位于[0, num_epochs)之间")
    if min_learning_rate <= 0:
        raise ValueError("min_learning_rate 必须大于0")

    base_learning_rates = [group["lr"] for group in optimizer.param_groups]
    if not base_learning_rates or any(
        learning_rate <= 0 for learning_rate in base_learning_rates
    ):
        raise ValueError("优化器的基础学习率必须大于0")
    if any(
        min_learning_rate > learning_rate
        for learning_rate in base_learning_rates
    ):
        raise ValueError("min_learning_rate 不能大于基础学习率")

    cosine_epochs = num_epochs - warmup_epochs

    def learning_rate_factor(epoch):
        # 调度器创建后，第一轮训练就使用 Warmup 的起始学习率。
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs

        # 让最后一个实际训练 epoch 恰好使用最小学习率。
        cosine_position = epoch - warmup_epochs
        cosine_denominator = max(1, cosine_epochs - 1)
        cosine_progress = min(1.0, cosine_position / cosine_denominator)
        cosine_value = 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )

        # 当前 AdamW 两个参数组使用相同基础学习率，因此取第一个组计算倍率。
        minimum_factor = min_learning_rate / base_learning_rates[0]
        return minimum_factor + (1.0 - minimum_factor) * cosine_value

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=learning_rate_factor,
    )


def set_random_seed(random_seed=RANDOM_SEED):
    """固定 CPU 和 CUDA 随机种子，使实验更容易复现。"""
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
    scaler,
    epoch,
    log_interval=LOG_INTERVAL,
    max_grad_norm=MAX_GRAD_NORM,
    mixup_alpha=MIXUP_ALPHA,
    cutmix_alpha=CUTMIX_ALPHA,
    cutmix_probability=CUTMIX_PROBABILITY,
):
    """完成一个 epoch 的训练，并返回平均损失和混合标签准确率。"""
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm 必须大于0")

    model.train()
    total_loss = 0.0
    total_weighted_correct = 0.0
    total_samples = 0
    amp_enabled = scaler.is_enabled()
    method_counts = {"mixup": 0, "cutmix": 0, "none": 0}

    for batch_index, (images, labels) in enumerate(data_loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        (
            mixed_images,
            labels_a,
            labels_b,
            mixing_lambda,
            mixing_method,
        ) = apply_mixup_or_cutmix(
            images=images,
            labels=labels,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            cutmix_probability=cutmix_probability,
        )
        method_counts[mixing_method] += 1

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(mixed_images)
            loss = calculate_mixed_loss(
                criterion=criterion,
                logits=logits,
                labels_a=labels_a,
                labels_b=labels_b,
                mixing_lambda=mixing_lambda,
            )

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        predictions = logits.argmax(dim=1)
        weighted_correct = (
            mixing_lambda * (predictions == labels_a).float()
            + (1.0 - mixing_lambda) * (predictions == labels_b).float()
        )
        batch_size = labels.shape[0]
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_weighted_correct += weighted_correct.sum().item()

        if log_interval > 0 and (
            batch_index % log_interval == 0
            or batch_index == len(data_loader)
        ):
            print(
                f"Epoch {epoch:03d} | "
                f"batch {batch_index:04d}/{len(data_loader):04d} | "
                f"loss {total_loss / total_samples:.4f} | "
                "mixed acc "
                f"{100.0 * total_weighted_correct / total_samples:.2f}%"
            )

    if total_samples == 0:
        raise RuntimeError("训练集为空，无法完成一个 epoch")

    return {
        "loss": total_loss / total_samples,
        "weighted_accuracy": total_weighted_correct / total_samples,
        "mixup_batches": method_counts["mixup"],
        "cutmix_batches": method_counts["cutmix"],
        "plain_batches": method_counts["none"],
    }


@torch.inference_mode()
def evaluate(model, data_loader, criterion, device, use_amp=True):
    """在不修改参数的情况下计算验证集或测试集的损失与准确率。"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    amp_enabled = use_amp and device.type == "cuda"

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = labels.shape[0]
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()

    if total_samples == 0:
        raise RuntimeError("评估数据集为空")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def fit(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=NUM_EPOCHS,
    checkpoint_path=BEST_MODEL_PATH,
    history_path=TRAINING_HISTORY_PATH,
    use_amp=True,
    log_interval=LOG_INTERVAL,
    max_grad_norm=MAX_GRAD_NORM,
    mixup_alpha=MIXUP_ALPHA,
    cutmix_alpha=CUTMIX_ALPHA,
    cutmix_probability=CUTMIX_PROBABILITY,
):
    """训练网络，并只保存验证集表现最好的 checkpoint。"""
    if num_epochs <= 0:
        raise ValueError("num_epochs 必须大于0")

    checkpoint_path = Path(checkpoint_path)
    history_path = Path(history_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = []
    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            epoch=epoch,
            log_interval=log_interval,
            max_grad_norm=max_grad_norm,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            cutmix_probability=cutmix_probability,
        )
        validation_metrics = evaluate(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        validation_accuracy = validation_metrics["accuracy"]
        validation_loss = validation_metrics["loss"]
        is_best = (
            validation_accuracy > best_validation_accuracy
            or (
                validation_accuracy == best_validation_accuracy
                and validation_loss < best_validation_loss
            )
        )

        # 本轮使用完当前学习率后再更新，新的学习率供下一轮训练使用。
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"],
            "train_weighted_accuracy": train_metrics["weighted_accuracy"],
            "mixup_batches": train_metrics["mixup_batches"],
            "cutmix_batches": train_metrics["cutmix_batches"],
            "plain_batches": train_metrics["plain_batches"],
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(epoch_record)

        with history_path.open("w", encoding="utf-8") as history_file:
            json.dump(history, history_file, ensure_ascii=False, indent=2)

        print(
            f"Epoch {epoch:03d}/{num_epochs:03d} 完成 | "
            f"lr {learning_rate:.8f} | "
            f"train loss {train_metrics['loss']:.4f}, "
            "mixed acc "
            f"{train_metrics['weighted_accuracy'] * 100:.2f}% | "
            f"val loss {validation_loss:.4f}, "
            f"acc {validation_accuracy * 100:.2f}%"
        )

        if is_best:
            best_validation_accuracy = validation_accuracy
            best_validation_loss = validation_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "training_config": {
                        "num_epochs": num_epochs,
                        "mixup_alpha": mixup_alpha,
                        "cutmix_alpha": cutmix_alpha,
                        "cutmix_probability": cutmix_probability,
                        "max_grad_norm": max_grad_norm,
                        "amp_enabled": amp_enabled,
                    },
                },
                checkpoint_path,
            )
            print(
                f"已保存新的最佳模型：{checkpoint_path} | "
                f"val acc={validation_accuracy * 100:.2f}%, "
                f"val loss={validation_loss:.4f}"
            )

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "best_validation_loss": best_validation_loss,
    }
