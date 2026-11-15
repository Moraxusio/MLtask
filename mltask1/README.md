# 任务一：实际提交对应的单文件方案

仅保留 `solution.py`（完整训练 → OOF 评估 → 测试预测入口）及本说明。`t1.py` 与精简前的 `solution.py` 去除 docstring 后 AST 相同，已删除重复副本；`dataset.py`、`model.py`、`train.py`、`predict.py` 属于旧流程，不是此入口的依赖，亦不保留。不打包 CSV、数据、权重、日志或历史评估 JSON。

## 与实际提交的对应关系

已只读核对 `本地项目目录/t1.csv` 与 `数据目录/submission-for-task2.csv`：两者 **11,273 bytes，逐字节一致**，SHA-256 均为：

```text
b8b533ac895453ab470f904cdb7a54e045cf936c6a5ac0609ca8fd4078b6aec9
```

原项目 `cv_report.json`、`logs/run_latest.log` 记录的最终方案为：ImageNet 预训练 **ConvNeXt-Tiny，384×384，5 折分层 CV，每折固定 30 epoch，EMA=0.99，Mixup=0，四视图 TTA，五折概率平均**。模型进行全网络微调，不是仅训练分类头；取最后一轮 EMA，而非按验证分数挑最佳 epoch。

这是“现有提交字节匹配 + 原配置、日志与实现对应”，不是本次独立重算预测，也不保证重训字节一致。原日志没有记录具体使用 `solution.py` 还是等价的 `t1.py` 入口。

源码现已删除 ResNet-50 备选模型、其参数分组分支及旧 `--quick` 简略运行。`--arch` 仅接受 `convnext_tiny`；默认保留最终 384、30 epoch、5 折、EMA=0.99、noMixup 的数学路径。EMA 开关与 Mixup 逻辑仍保留，供报告第一章已有的真实消融使用，不代表最终方案启用了 Mixup。超参数与输出路径仍可配置；未增加其他模型实现。EMA 的新模型构造/加载顺序也未改成 deepcopy，以免改变 RNG。

## 数据、依赖与运行目录

数据由使用者自行准备，不复制进源码包。当前机器的只读数据根目录是：

```text
数据目录/dataset-for-task2/dataset-for-task2/
  train/
    Black-grass/*.png
    Common wheat/*.png
    Loose Silky-bent/*.png
    Scentless Mayweed/*.png
    Sugar beet/*.png
  test/*.png
```

训练图共 500 张（每类 100），测试图共 378 张。目录名中的 `task2` 沿用原命名，不是另一任务。提交列名为 `ID,Category`，ID 为测试 PNG 文件名，类别顺序如上。

直接第三方依赖：`numpy`、`pandas`、`torch`、`torchvision`、`Pillow`、`scikit-learn`。应使用相互兼容的 PyTorch/torchvision 环境；原训练版本未锁定，此处不虚构版本。模型使用 `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`，需能下载 torchvision 官方预训练权重或提前准备缓存。程序自动选 CUDA（使用 bfloat16 autocast），否则 CPU；无 `--device` 参数。原日志只记录 CUDA，没有 GPU 型号/库版本。

## 实际配置对应命令（未执行）

下面命令使用当前机器原项目的只读数据，输出写入源码包任务目录，不覆盖原项目。其他机器请替换 `cd` 和 `--data-root` 的绝对路径；所有相对输出路径相对于工作目录。运行会真正训练并推理，且覆盖同名输出文件。

```bash
cd 本地项目目录/github_code/mltask1
python3 solution.py \
  --data-root 数据目录/dataset-for-task2/dataset-for-task2 \
  --arch convnext_tiny \
  --img-size 384 \
  --epochs 30 \
  --batch-size 24 \
  --folds 5 \
  --head-lr 0.0005 \
  --backbone-lr 0.00005 \
  --weight-decay 0.05 \
  --label-smoothing 0.1 \
  --mixup-alpha 0.0 \
  --ema-decay 0.99 \
  --seed 42 \
  --workers 8 \
  --log-every 5 \
  --out submission-for-task2.csv \
  --report cv_report.json
```

参数已按源码 argparse 声明核实。这是依据现存配置重建的命令，不是已找回的历史启动命令；除数据根目录改成绝对路径外，对应原 JSON 的全部配置。故意不传 `--no-mixup`，使其布尔值保持历史记录的 `false`，而用 `--mixup-alpha 0.0` 关闭 Mixup。旧 `quick=false` 字段不再出现在新报告中，`--quick` 已不支持；这不改变实际训练数学路径。

**没有 `--tta` / `--tta4` 参数。** 程序分别计算 `none` 与 `tta4` 的 OOF accuracy，再自动选择较高者（同分时按当前顺序选 `none`）。原日志是 `none=0.9280`、`tta4=0.9300`，因此选中 `tta4`。四视图为原图、水平翻转、垂直翻转、180°旋转；先平均视图 softmax 概率，再平均五折概率。重新训练可能选出不同方案；为保持原算法，未新增强制 TTA 开关。

预期生成文件（运行时生成，不纳入源码仓库）：

- `本地项目目录/github_code/mltask1/submission-for-task2.csv`：378 条预测，表头 `ID,Category`。
- `本地项目目录/github_code/mltask1/cv_report.json`：本次参数、OOF 分数、选中 TTA、混淆矩阵及类别。
- 训练摘要只打印到标准输出；此命令不生成日志文件。每折预测后释放模型，不保存 checkpoint 或逐样本概率。不会自动生成名为 `t1.csv` 的文件；如需此名称，可将 `--out` 改为 `t1.csv`，但新预测不能冒充已经核验的原提交。

## 复现边界

- 原项目未保存最终模型权重及 OOF/test 概率，无法不重训而独立重算当前提交；CSV 只含最终标签，不能还原模型和概率。源码没有独立加载最终权重进行推理的模式。
- 未记录完整历史命令、依赖锁定或硬件环境；固定 seed 不保证跨环境数值或输出字节相同。删除未用分支后仍保留实际路径的算法与默认值。标签平滑注释已纠正：5 类、ε=0.1 时真类为 0.92，其余各 0.02。
- 0.9300 是开发过程中的 OOF accuracy，不是无标签测试集准确率。EMA/Mixup/TTA 的选择使用了 OOF，仍有选择偏差。
- 此前提交对应核验包括文件字节、源码 AST、依赖及 CLI；本次分支精简另做逐函数 AST、隔离 argparse、符号/调用签名及合成标量公式检查，见 代码版本说明。未导入训练依赖、未运行项目入口、未训练/推理、未下载权重。
