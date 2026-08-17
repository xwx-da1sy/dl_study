"""Tiny ViT 的损失函数、优化器与学习率调度器。"""

from pathlib import Path

import torch
from torch import nn

from .config import (
    BEST_MODEL_PATH,
    LEARNING_RATE,
    LOG_INTERVAL,
    MAX_GRAD_NORM,
    MIN_LEARNING_RATE,
    NUM_EPOCHS,
    RANDOM_SEED,
    WEIGHT_DECAY,
)


def create_training_components(
    model,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    num_epochs=NUM_EPOCHS,
    min_learning_rate=MIN_LEARNING_RATE,
):
    """
    作用：
        创建 Tiny ViT 训练所需的交叉熵损失、AdamW 优化器和余弦退火调度器。

    参数：
        model：需要训练的完整 TinyViT 模型。
        learning_rate：AdamW 的初始学习率，当前为 3e-4。
        weight_decay：AdamW 的权重衰减系数，当前为 0.05。
        num_epochs：计划训练的总 epoch 数，也作为余弦退火周期，当前为 100。
        min_learning_rate：余弦退火允许降低到的最小学习率，当前为 1e-6。

    返回值：
        criterion：CrossEntropyLoss，多分类交叉熵损失函数。
        optimizer：AdamW 优化器。
        scheduler：CosineAnnealingLR 余弦退火学习率调度器。
    """
    if num_epochs <= 0:
        raise ValueError("num_epochs 必须大于 0")

    # nn.CrossEntropyLoss
    # 作用：计算单标签多分类交叉熵，适用于 CIFAR-10 的10分类任务。
    # 输入参数：logits 的 shape 为 B x 10；labels 的 shape 为 B、类型为 int64。
    # 返回值：一个标量 loss。
    # 注意：它内部已包含 LogSoftmax，因此分类头后面不需要手动添加 Softmax。
    criterion = nn.CrossEntropyLoss()

    # torch.optim.AdamW
    # 作用：根据梯度更新模型参数，并以解耦方式执行权重衰减。
    # 参数：
    #   model.parameters()：完整 Tiny ViT 的所有可训练参数；
    #   lr=3e-4：初始学习率；
    #   weight_decay=0.05：权重衰减强度。
    # 返回值：AdamW 优化器对象。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # torch.optim.lr_scheduler.CosineAnnealingLR
    # 作用：按照余弦曲线把学习率从初始值平滑降低到最小值。
    # 参数：
    #   optimizer：需要调整学习率的 AdamW；
    #   T_max=100：完成一次退火所需的 scheduler.step() 次数；
    #   eta_min=1e-6：最低学习率。
    # 返回值：余弦退火学习率调度器。
    # 后续训练时应在每个 epoch 结束后调用一次 scheduler.step()。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=num_epochs,
        eta_min=min_learning_rate,
    )

    return criterion, optimizer, scheduler


def set_random_seed(random_seed=RANDOM_SEED):
    """
    作用：固定 PyTorch 的随机种子，使数据划分和模型初始化更容易复现。

    参数：
        random_seed：随机种子，当前默认为 42。

    返回值：无；直接设置 CPU 和 CUDA 随机数生成器的状态。
    """
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
):
    """
    作用：完成一个 epoch 的训练，包括前向、损失、反向传播和参数更新。

    参数：
        model：需要训练的 TinyViT。
        data_loader：训练集 DataLoader。
        criterion：CrossEntropyLoss。
        optimizer：AdamW。
        device：CPU 或 CUDA 设备。
        scaler：CUDA 混合精度使用的 GradScaler；关闭混合精度时它处于禁用状态。
        epoch：当前 epoch 编号，仅用于打印进度。
        log_interval：每隔多少个 batch 打印一次训练信息。
        max_grad_norm：梯度范数上限，当前为 1.0。

    返回值：
        average_loss：当前 epoch 的样本平均损失。
        accuracy：当前 epoch 的训练准确率，范围为 0～1。
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    amp_enabled = scaler.is_enabled()

    for batch_index, (images, labels) in enumerate(data_loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # set_to_none=True 不把旧梯度写成 0，而是设为 None，可以减少内存写入。
        optimizer.zero_grad(set_to_none=True)

        # CUDA 上使用自动混合精度；CPU 上 enabled=False，会正常使用 float32。
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        if amp_enabled:
            # 先放大 loss 再反向传播，降低 float16 梯度下溢的风险。
            scaler.scale(loss).backward()

            # 梯度裁剪前必须先还原真实梯度大小。
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
    """
    作用：在不计算梯度的情况下评估验证集或测试集。

    参数：
        model：需要评估的 TinyViT。
        data_loader：验证集或测试集 DataLoader。
        criterion：CrossEntropyLoss。
        device：CPU 或 CUDA 设备。
        use_amp：CUDA 上是否使用自动混合精度。

    返回值：
        average_loss：整个数据集的样本平均损失。
        accuracy：整个数据集的分类准确率，范围为 0～1。
    """
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
    use_amp=True,
    log_interval=LOG_INTERVAL,
    max_grad_norm=MAX_GRAD_NORM,
):
    """
    作用：
        依次执行多个训练 epoch，每轮结束后验证并更新余弦退火学习率，
        同时保存验证准确率最高的模型检查点。

    参数：
        model：完整 TinyViT。
        train_loader：训练集 DataLoader。
        validation_loader：验证集 DataLoader。
        criterion：CrossEntropyLoss。
        optimizer：AdamW。
        scheduler：CosineAnnealingLR，每个 epoch 结束后更新一次。
        device：CPU 或 CUDA 设备。
        num_epochs：训练总轮数。
        checkpoint_path：最佳模型保存路径；传入 None 可以关闭保存。
        use_amp：CUDA 上是否启用混合精度。
        log_interval：训练进度打印间隔。
        max_grad_norm：梯度裁剪上限。

    返回值：
        history：每个 epoch 的 loss、accuracy 和 learning_rate 记录列表。
        best_validation_accuracy：训练期间最高验证准确率。
    """
    if num_epochs <= 0:
        raise ValueError("num_epochs 必须大于 0")

    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = []
    best_validation_accuracy = -1.0

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

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
        )
        validation_loss, validation_accuracy = evaluate(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        # 余弦退火每个 epoch 只更新一次，供下一轮训练使用。
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

        print(
            f"Epoch {epoch:03d}/{num_epochs:03d} 完成 | "
            f"lr {learning_rate:.8f} | "
            f"train loss {train_loss:.4f}, acc {train_accuracy * 100:.2f}% | "
            f"val loss {validation_loss:.4f}, acc {validation_accuracy * 100:.2f}%"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            if checkpoint_path is not None:
                # 只保存最佳模型，同时保留优化器和调度器状态，方便以后恢复训练。
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
                print(f"已保存新的最佳模型：{checkpoint_path}")

    return history, best_validation_accuracy
