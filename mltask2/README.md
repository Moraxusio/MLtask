# 任务二：实际提交的最小生成链

本目录对应实际上传的 `t2.csv`：**三种 CNN 五折固定融合，再按训练图像重复查表修正**，不是后来单模型或 v4 提交。现存 15 个历史折预测已重建出全部 7065 行，CSV 120116 字节，SHA-256：

```text
b3e32c3a9d763cc43e806e53fbf69b07a8bec494997a8839c05b2e50014c27e2
```

该哈希只是外部核验结果，不是生成器的输入或预置答案。训练、权重、图像及预测缓存未打包；没有宣称重新训练可逐字节再现历史权重。

## 最小文件与依赖

| 文件 | 用途 |
|---|---|
| `src/audit.py` | 从训练/测试图像构建像素缓存、训练标签、索引、完全重复分组 |
| `src/split.py` | 原始五折 StratifiedGroupKFold，使用完全重复分组 |
| `src/common.py` | 三种实际贡献模型、训练增强、EMA、预处理与数据加载 |
| `src/train.py` | 单模型单折训练，按验证集 plain accuracy 选最佳 EMA，保存 OOF/测试 plain 与翻转 TTA 概率 |
| `src/reproduce_actual.py` | 固定历史融合、训练重复查表、CSV 导出和标准输出审计信息；仅需 NumPy |

`audit.py`、`split.py` 保持原样。`train.py` 删除历史实际模型未使用的 STN/SE 包装与其它 split 分支；`common.py` 删除无贡献架构/预训练别名。保留原 CNN 分支的增强、优化器、学习率、随机种子、EMA、最佳轮次选择、推理预处理和概率保存算法。

未保留动态 `select.py` 搜索（固定版本取代其融合部分）、per-class/v2、STN/DINO、探测器、单模型包及单模型导出。原 `train.py` **无条件导入** STN 支持文件，即使训练不用 STN 也有模块依赖；此处连同未用分支一起移除。删除 `select.py` 也消除了同名模块遮蔽 Python 标准库的问题，不再需要特殊启动器。

## 从已有历史 artifacts 生成（本次已执行）

从本目录运行，`ARTIFACTS` 由使用者指定到原始缓存目录，避免在原工程上训练或改写缓存：

```bash
export ARTIFACTS=/path/to/original/artifacts
python3 -B src/reproduce_actual.py \
  --artifacts "$ARTIFACTS" \
  --out ../../生成的 t2.csv
```

在当前交付目录布局下，唯一写出的文件是 `生成的 t2.csv`；所有输入只读，报告以 JSON 打印到标准输出，不生成中间概率或报告文件。可自行指定其它外部 CSV 输出位置，但不能写进输入 artifacts。生成脚本**没有模板、提交 CSV 或测试答案输入参数**。

需要如下输入（20 个文件）：

- `file_index.json`、`train_pixels.npy`、`test_pixels.npy`、`train_labels.npy`、`fold.npy`；像素来自 `audit.py` 的 48×48 灰度 uint8 图像解码，训练标签来自类别目录。
- `p_cnt_224_30_fold{0..4}_preds.npz`。
- `p_r34_224_30_fold{0..4}_preds.npz`。
- `eb0_224_30_fold{0..4}_preds.npz`。

NPZ 应包含 `va_idx, va_plain, va_tta, te_plain, te_tta`。生成器检查每折验证索引与 `fold.npy` 一致、OOF 恰好覆盖一次、概率尺寸/有限值/归一化、像素类型、标签与训练目录一致、类别顺序及测试 ID 顺序，打印每个输入 SHA-256。**不要读取或替换为 `test_probs_selected.npy`**：该同名文件已被 v4 覆盖，与这里的历史融合有 345 行 argmax 不同。固定 tag/视图避免取错融合版本；历史输入的完整哈希证据见交付中的 任务二实验记录。NPZ 本身不保存测试 ID、源码版本或图像哈希，因此仅靠形状检查不能鉴定任意外来缓存，严格历史复算还需核对输入哈希，不能混用重排索引或不同训练的文件。

### 固定算法

| 模型 tag | 架构 | 五折内测试视图 | 融合权重 |
|---|---|---|---:|
| `p_cnt_224_30` | ConvNeXt-Tiny | `te_tta`（原图与水平翻转均值） | 0.50 |
| `p_r34_224_30` | ResNet34 | `te_tta` | 0.25 |
| `eb0_224_30` | EfficientNet-B0 | `te_plain` | 0.25 |

先按 fold 0→4、float64 累加并取算术均值，再按历史选择顺序 Tiny→R34→Tiny→EB0 计算 `exp(sum(log(p + 1e-9))/4)`；OOF 使用同样视图和融合。先验指数为 0，不进行额外归一化。参数由旧 `selection.json` 确认，不在新脚本中再次搜索。

随后为每个训练图像的原始灰度像素建立 MD5→训练行索引表。测试图像匹配时，仅当所有匹配训练标签一致且票数至少为 1 时用该标签替换预测；无匹配或训练标签冲突则保留模型预测。该规则从原 `dup_override.py` 提取，未硬编码测试样本、修改行数或标签。CSV 使用 `file_index.json` 中原先按文件名排序的测试 ID，表头 `ID,Emotion`，0–5 整数类别、UTF-8 无 BOM、LF 换行且末行有换行。

类别顺序：`Angry, Fear, Happy, Sad, Surprise, Neutral`。

## 从原始图像训练到折概率（命令已核对，未重新执行）

准备以下目录；只能在整理副本或新工作目录中运行，不能指向只读原项目：

```text
mltask2/
  src/
  fer_data/fer_data/
    train/{Angry,Fear,Happy,Sad,Surprise,Neutral}/*.jpg
    test/*.jpg
  weights/
    convnext_tiny.fb_in22k_ft_in1k.safetensors
    resnet34.a1_in1k.safetensors
    tf_efficientnet_b0.ns_jft_in1k.safetensors
  artifacts/      # audit/split/train 创建；勿混用不同训练版本
  logs/          # train 创建
```

已有缓存复算仅需 `numpy`。审计增加 `Pillow`，分折增加 `scikit-learn`；训练还需 CUDA 版 `torch`、兼容的 `torchvision`、`timm`、`safetensors`。先按本机 CUDA 安装匹配的 PyTorch/torchvision，再安装其它依赖。此次 CPU 复算实际使用 Python 3.10.12、NumPy 2.2.6；这些不是历史训练环境锁定文件。

原 15 份训练 summary 与 15 份 JSONL 日志确认：224 输入、30 epochs、Tiny batch 96，其余 batch 128，原 `fold.npy`；所有日志含 epoch 0–29。其余共同参数：seed 20261011（训练种子加 fold）、workers 6、AdamW lr 0.0003、head lr ×10、weight decay 0.05、warmup 0.1 后 cosine、label smoothing 0.1、mixup 0.2、cutmix 1.0、mix probability 0.5、EMA 0.999、drop 0.2、drop path 0.1、梯度裁剪 1.0、完整训练 batch。

```bash
python3 -B src/audit.py
python3 -B src/split.py
for fold in 0 1 2 3 4; do
  python3 -B src/train.py --arch convnext_tiny --fold "$fold" \
    --img-size 224 --epochs 30 --batch-size 96 --tag p_cnt_224_30
  python3 -B src/train.py --arch resnet34 --fold "$fold" \
    --img-size 224 --epochs 30 --batch-size 128 --tag p_r34_224_30
  python3 -B src/train.py --arch tf_efficientnet_b0 --fold "$fold" \
    --img-size 224 --epochs 30 --batch-size 128 --tag eb0_224_30
done
python3 -B src/reproduce_actual.py --artifacts artifacts \
  --out ../../生成的 t2.csv
```

训练脚本默认输入尺寸/epochs 仍是原接口的 112/14，必须显式使用上面的 224/30。图像增广保留原配置：翻转、旋转 ±12°、平移 0.10、缩放 0.12、shear 6°、亮度/对比度 0.25、擦除概率 0.35、最大面积 0.20。模型输入为双三次上采样、灰度复制三通道、ImageNet 归一化；CUDA BF16 autocast，cuDNN benchmark 开启。保留原半精度保存 EMA 检查点、重新载入 float32 后最终预测的行为。

## 复核结果与边界

- 原始图像重新解码后，与 28275 个训练像素、7065 个测试像素缓存及全部训练标签完全一致；未读取 hint 或外部测试答案。
- 新融合与原 `select.collect`/历史 log 公式的完整 OOF/测试数组精确一致；原 `dup_override.main` 在内存接入重建概率后，产生同一完整 CSV。
- 融合训练 OOF accuracy **0.7158620689655173**，macro-F1 **0.6981250888506717**；视图/配方曾使用同一 OOF 选择，存在选择偏差，不是独立评估。
- 测试中 520 行有训练像素重复：511 行标签一致，其中 475 行原预测已同意、36 行被改写；9 行冲突跳过。训练集共有 39 个重复标签冲突组。这里的 36 是复算结果，生成代码未固定该数。
- 后处理依赖训练与测试完全重复，**不是独立泛化增益**。训练标签一致不代表真实测试标签必然正确；没有测试精度或“提升 36/7065”的实测结论。不对规则许可作判断。
- 代码支持从数据和模型权重开始完成训练、验证、融合与提交生成。
- 未纳入历史精确环境锁、CUDA/cuDNN/GPU 信息及预训练权重的原始下载版本/校验记录；权重、数据与大体积 artifacts 未随代码分发。新训练不是已验证的逐字节复现。完整日志/配置仍在原只读项目，本目录仅保留实际生成链所需代码。
