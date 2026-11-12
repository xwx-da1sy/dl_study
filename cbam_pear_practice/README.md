# 梨叶分类：MobileNetV2 与 CBAM 对照

目标：参照论文比较预训练 MobileNetV2 加入 CBAM 前后的四类梨叶分类效果。原论文：[Improving Deep Learning-based Plant Disease Classification with Attention Mechanism](https://doi.org/10.1007/s10343-022-00796-y)。

## 数据准备

数据集来源：[DiaMOS Plant](https://zenodo.org/records/5557313)，原始 `Pear.zip` 约 13.1 GB。将完整压缩包放到 `data/Pear.zip`，然后运行：

```powershell
python cbam_pear_practice/prepare_data.py
```

脚本检查作者提供的 MD5，仅将四类叶片图片转换为 256×256 JPEG 缓存，并检查数量分别为 43、884、54、2025。缓存会重新编码，属于近似实现；训练时再缩放为 224×224。图像缓存和原始包均在 `data/`，不会加入 Git。

## 训练

```powershell
python cbam_pear_practice/train_compare.py
```

两组使用相同的第一折分层测试集、训练内验证集、预训练骨干、分类头初始化、增强策略和训练超参数；两个模型均冻结 MobileNetV2 特征提取器和其 BN 运行统计。仅实验组在输出特征后加入教学版 CBAM，分类头由 BN、256 维全连接、ReLU、Dropout 和四分类层组成。

默认最多 100 轮、batch size 32、Adam 学习率 2e-5；验证损失停滞时降学习率并在最低 25 轮后可早停。根据验证准确率选模型，测试折只在训练结束后评估。保存每折拆分、训练记录、各类精确率/召回率/F1、混淆矩阵和模型参数。结果写入 `results/`，重复运行覆盖同名文件。

这是一折探索实验，独立内部验证集、缓存重编码、分类头和增强细节不保证与论文一致；不能称为论文十折的精确复现。类别很不均衡，少数类测试样本极少，应结合各类指标判断且避免将百分比差异过度解读。

## 官方数据小样本试跑

`prepare_pilot.py` 通过 HTTP Range 读取官方 ZIP 的目录，在确认四类原始数量后，每类以固定随机种子随机选取 40 张，并对每张图片进行 ZIP CRC 校验。该试跑共 160 张，类别经过人为均衡，不能与原论文或完整数据集的结果直接比较。

```powershell
python cbam_pear_practice/prepare_pilot.py
python cbam_pear_practice/train_compare.py --pilot
```

试跑先划分一折测试（每类 4 张），再从余下部分划出验证集；少数样本导致指标波动很大。输出在 `pilot_results/`，重复执行会覆盖。完整实验仍应使用原始包及 `prepare_data.py`。
