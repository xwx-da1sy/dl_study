"""Tiny ViT 的损失函数、优化器、学习率调度器与 Early Stopping。"""

import json
from pathlib import Path

import torch
from torch import nn

from .config import (
    BEST_MODEL_PATH,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    LOG_INTERVAL,
    MAX_GRAD_NORM,
    MIN_LEARNING_RATE,
    MIXUP_ALPHA,
    NUM_EPOCHS,
    RANDOM_SEED,
    TRAINING_HISTORY_PATH,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)


def create_training_components(
    model,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    num_epochs=NUM_EPOCHS,
    min_learning_rate=MIN_LEARNING_RATE,
    label_smoothing=LABEL_SMOOTHING,
    warmup_epochs=WARMUP_EPOCHS,
):
    """
    创建 Tiny ViT 训练所需组件。

    与原版本相比：
        1. CrossEntropyLoss 加入 label_smoothing=0.1；
        2. 前 warmup_epochs 轮线性升温，之后使用余弦退火。
    """
    if num_epochs <= 0:
        raise ValueError("num_epochs 必须大于 0")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing 必须位于 [0, 1) 区间")
    if warmup_epochs < 0 or warmup_epochs >= num_epochs:
        raise ValueError("warmup_epochs 必须位于 [0, num_epochs) 区间")

    # Label Smoothing 可以降低模型对训练标签的过度自信，从而改善泛化。
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # warmup 后进入余弦退火；LambdaLR 以初始学习率为基准计算每轮倍率。
    def learning_rate_factor(epoch):
        if warmup_epochs and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        cosine_epoch = epoch - warmup_epochs
        cosine_total = max(1, num_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(
            torch.pi * cosine_epoch / cosine_total
        )).item())
        minimum_factor = min_learning_rate / learning_rate
        return minimum_factor + (1.0 - minimum_factor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=learning_rate_factor,
    )

    return criterion, optimizer, scheduler


def set_random_seed(random_seed=RANDOM_SEED):
    """固定 PyTorch 随机种子，使数据划分和模型初始化更容易复现。"""
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
):
    """完成一个 epoch 的训练。"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    amp_enabled = scaler.is_enabled()

    if mixup_alpha < 0:
        raise ValueError("mixup_alpha 不能为负数")

    for batch_index, (images, labels) in enumerate(data_loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixup 只作用于训练 batch；验证和测试仍使用原始标签。
        if mixup_alpha > 0:
            mixup_lambda = torch.distributions.Beta(
                mixup_alpha, mixup_alpha
            ).sample().item()
            permutation = torch.randperm(images.size(0), device=device)
            mixed_images = (
                mixup_lambda * images
                + (1.0 - mixup_lambda) * images[permutation]
            )
            labels_a = labels
            labels_b = labels[permutation]
        else:
            mixup_lambda = 1.0
            mixed_images = images
            labels_a = labels_b = labels

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(mixed_images)
            loss = (
                mixup_lambda * criterion(logits, labels_a)
                + (1.0 - mixup_lambda) * criterion(logits, labels_b)
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

        batch_size = labels.shape[0]
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()

        if log_interval > 0 and (
            batch_index % log_interval == 0 or batch_index == len(data_loader)
        ):
            print(
                f"Epoch {epoch:03d} | "
                f"batch {batch_index:04d}/{len(data_loader):04d} | "
                f"loss {total_loss / total_samples:.4f} | "
                f"acc {100.0 * total_correct / total_samples:.2f}%"
            )

    if total_samples == 0:
        raise RuntimeError("训练集为空，无法完成一个 epoch")

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return average_loss, accuracy


@torch.inference_mode()
def evaluate(model, data_loader, criterion, device, use_amp=True):
    """在不计算梯度的情况下评估验证集或测试集。"""
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

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return average_loss, accuracy


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
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
):
    """
    训练 Tiny ViT，并根据 validation loss 保存最佳模型和执行 Early Stopping。

    设计原则：
        - checkpoint 使用 validation loss，而不是 validation accuracy；
          这是因为本次实验中 validation accuracy 后期基本持平，
          但 validation loss 已明显上升，更能反映过拟合和过度自信。
        - validation loss 连续 patience 个 epoch 没有至少 min_delta 的改善时停止。

    返回值保持与原版本兼容：
        history：每个 epoch 的训练记录；
        best_validation_accuracy：训练期间出现过的最高验证准确率。
    """
    if num_epochs <= 0:
        raise ValueError("num_epochs 必须大于 0")
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience 必须为正整数或 None")
    if early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta 不能为负数")

    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = []
    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if history_path is not None:
        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]

        train_loss, train_accuracy = train_one_epoch(
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
        )

        validation_loss, validation_accuracy = evaluate(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        # 先保存旧值，后面用它判断本轮是否刷新最佳准确率。
        previous_best_validation_accuracy = best_validation_accuracy
        best_validation_accuracy = max(
            best_validation_accuracy, validation_accuracy
        )

        # 当前 epoch 使用完当前学习率后再 step，供下一轮训练使用。
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(epoch_record)

        if history_path is not None:
            with history_path.open("w", encoding="utf-8") as history_file:
                json.dump(history, history_file, ensure_ascii=False, indent=2)

        print(
            f"Epoch {epoch:03d}/{num_epochs:03d} 完成 | "
            f"lr {learning_rate:.8f} | "
            f"train loss {train_loss:.4f}, acc {train_accuracy * 100:.2f}% | "
            f"val loss {validation_loss:.4f}, acc {validation_accuracy * 100:.2f}%"
        )

        # 只有 validation loss 至少下降 min_delta 才认为出现了实质改善。
        validation_accuracy_improved = (
            validation_accuracy > previous_best_validation_accuracy
        )
        validation_loss_improved = (
            validation_loss < best_validation_loss - early_stopping_min_delta
        )

        # 以 accuracy 作为最终模型选择标准；loss 仅用于 early stopping。
        if validation_accuracy_improved or (
            validation_accuracy == best_validation_accuracy
            and validation_loss_improved
        ):
            best_validation_loss = validation_loss
            epochs_without_improvement = 0

            if checkpoint_path is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "validation_loss": validation_loss,
                        "validation_accuracy": validation_accuracy,
                    },
                    checkpoint_path,
                )
                print(
                    f"已保存新的最佳模型：{checkpoint_path} | "
                    f"val loss={validation_loss:.4f}, "
                    f"val acc={validation_accuracy * 100:.2f}%"
                )
        elif validation_loss_improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

            if early_stopping_patience is not None:
                print(
                    "Early Stopping 计数："
                    f"{epochs_without_improvement}/{early_stopping_patience}"
                )

                if epochs_without_improvement >= early_stopping_patience:
                    print(
                        f"验证 loss 已连续 {early_stopping_patience} 个 epoch "
                        "没有明显改善，提前停止训练。"
                    )
                    break

    return history, best_validation_accuracy
