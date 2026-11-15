# 任务四：实际上传 v4 的最小代码

本目录仅保留与实际 `t4.csv` 对应的训练、固定 v4 集成、OOF 评价和 RLE 转换链路，不保留 v1–v3 推理或未完成家族的搜索入口。源码证据见任务四实验记录，本次检查见 任务四实验记录。

## 实际版本与最小清单

实际上传文件与源项目 `segmentation/submission.csv`、`submissions/v4_unet+r34x2+eb3x2_th0.36.csv` 字节相同，SHA-256：

```text
ce0003e2eb0aae59073c95327b9215c87253d0a2de1378ccca4f59d88592c62a
```

| 必要文件 | 用途 / 依赖 |
|---|---|
| `train.py` | 基础四通道 U-Net、数据/增强/loss、四翻转 TTA、CV/全量训练；`train2.py` 复用 Block、Dataset、loss、数据读取等，不能删除 |
| `train2.py` | timm 编码器 U-Net 与 x2 缩放/TTA；生成两家族模型与 OOF |
| `predict.py` | **仅保留 `official_mask` helper**，供 v4 OOF 评价使用；已移除 v3 搜索/推理主流程 |
| `predict_v4.py` | 固定最终 35 模型路径、阈值与家族权重；预检、CPU OOF 评价、显式 CUDA 推理及 CSV 导出 |
| `segmentation/segmentation_to_csv.py` | 必要：PNG 不是最终提交；保留原 resize / >0 / RLE 算法，增加输入输出 CLI |

`official_mask` 与转换器 `turn_to_str` 相对源实现的函数 AST 完全相同，未作数学修改。训练函数/模型类保持原算法，仅修正说明及为 `train2.py` 创建 `work/`。固定入口取消自动家族扫描和阈值搜索；不再因额外的 r34、其他 seed 或不完整 checkpoint 改变实际提交配方。

### 固定配方及权重文件（全部必需）

路径相对 `work/`，按下列顺序累计概率：

- U-Net：`fold{0..4}_s{0..3}.pt`（seed 优先，共 20 个），加 `fold0_full_s{10..14}.pt`（5 个）。
- ResNet34×2：`fold{0..4}_r34x2_s0.pt`（5 个），编码器 `resnet34.a1_in1k`。
- EfficientNet-B3×2：`fold{0..4}_eb3x2_s0.pt`（5 个），编码器 `efficientnet_b3.ra2_in1k`。
- 家族内部平均，再三家族 **1:1:1 等权**，最后乘 FOV，阈值固定 **`>0.36`**；白色血管 255、黑色背景 0。
- 不是 35 模型简单平均：单个 U-Net 最终权重 1/75，其他两家族单模型各 1/15；每模型四翻转 TTA，缩放/精度/累加顺序沿用原实现。

默认只检查全部 35 个指定文件非空，不使用 glob，也不会用额外权重补足缺失项；推理严格匹配 state_dict。文件名/形状检查不证明文件内容就是历史权重，使用者仍须提供可信的同源文件，不能随意替换同名模型。

## 外部资产与依赖

数据、权重、概率数组、预测 PNG/CSV **不随代码入仓库**。训练入口默认基于脚本目录，不依赖启动目录：

```text
mltask4/
  segmentation/train/image/{1..20}.jpg
  segmentation/train/label/{1..20}.jpg
  segmentation/test/image/{1..20}.jpg
  pretrained/resnet34.a1_in1k.safetensors
  pretrained/efficientnet_b3.ra2_in1k.safetensors
  work/                         # 上述 35 个 .pt；评价另需 OOF
```

依赖为 NumPy、Pillow、pandas、torch、torchvision、timm、OpenCV。训练和模型推理要求 CUDA，预训练家族使用 BF16；已有 PNG 转换和 OOF 复算只需 CPU，无需下载预训练权重。当前检查环境不等于原训练环境，没有原环境锁文件。

## 固定最终生成路径

在本目录执行（`--work`、`--seg` 支持外部资产路径）：

```bash
# 默认安全预检：不训练、不推理、不评价，也不加载权重到 GPU
python3 predict_v4.py --check-only --work /path/to/task4/work

# 仅在明确需要实际模型推理时运行；输出目录必须为空或不存在
python3 predict_v4.py --infer \
  --work /path/to/task4/work --seg /path/to/task4/segmentation \
  --output-dir /path/to/new_v4_output
```

固定生成 `new_v4_output/image/{1..20}.png`、`submission.csv` 和 `final_v4.json`（实际权重文件清单、25/5/5 数量与固定配方）。无需另跑 converter；入口直接调用原转换函数。不会覆盖源项目提交、搜索阈值或读取上传 CSV。新推理导出 CSV 按数字 ID 1–20 排序；历史提交使用未排序的文件系统遍历顺序，故即使掩码一致，新推理 CSV 也可能仅行序不同而非字节相同。推理入口按照固定模型配置生成预测掩码与提交文件。

已有 PNG 可单独纯 CPU 转换（不需要 checkpoint）：

```bash
python3 segmentation/segmentation_to_csv.py \
  --image-dir /path/to/existing_v4_pngs --output /path/to/t4.csv
```

独立转换保留源目录遍历顺序；顺序可能跨文件系统变化。转换规则固定为灰度 → 512×512 双线性 → >0 → C 行优先展开 → 1-based `(start,length)` RLE，列名 `Id,Predicted`。不是列优先，也不是 resize 后 >127。

## OOF 评价（不训练、不推理）

需要 `work/oof_s{0..3}.npy` 及两个家族 `oof_{r34x2,eb3x2}_s0.npy`；每家族也可由 `partial_<tag>_s0_f{0..4}.npy` 按原 seed0 划分在内存组装。若完整缓存和五个 partial 同时存在，必须逐元素一致，否则报错；不回写源缓存。检查概率 shape、float32、有限值与范围。

```bash
python3 predict_v4.py --evaluate-oof \
  --work /path/to/task4/work --seg /path/to/task4/segmentation
```

固定阈值 .36，三家族等权，20 张训练图的逐图 Dice 均值：

| 口径 | 实测复算 |
|---|---:|
| 提交转换流程 OOF Dice | 0.8489799511585858 |
| 原尺寸 OOF Dice | 0.8159744669834493 |

这是已有 OOF 的离线代理指标，**不是隐藏测试分数**。历史使用同一 OOF 选择家族权重/阈值并报告最优分数，有选择偏差。五个全量 U-Net 只用于测试集成，没有对应 OOF；因此这也不是完整 35 模型系统的独立验证成绩。只支持本任务 20 张、584×565 原尺寸；没有患者分组或独立外部验证。

## 训练 CLI 与历史参数缺口

以下给出可直接执行的训练命令与参数说明。

1. s2/s3 的 JSON 明确保存 `folds=5,iters=6000,bs=16,crop=256,base=32,lr=.002`。s0/s1 的 JSON 没有 args；全量 s10–s14 也缺完整启动参数。日志/权重能确认四种子五折、五个全量模型、base32 和 6000 迭代，不能据此确认所有历史 batch/crop/lr。以下对缺失项采用当前实现默认值，属于重建方案，不是历史命令回放：

```bash
for seed in 0 1 2 3; do
  python3 train.py --seed "$seed" --folds 5 --iters 6000 \
    --bs 16 --crop 256 --base 32 --lr 0.002
done
for seed in 10 11 12 13 14; do
  python3 train.py --seed "$seed" --full --folds 5 --iters 6000 \
    --bs 16 --crop 256 --base 32 --lr 0.002
done
```

2. 两个 timm 家族完整五折权重真实存在；唯一保存的明确启动命令来自 eb3x2 fold4（省略机器 Python 绝对路径与日志重定向）：

```bash
python3 train2.py --seed 0 --only_fold 4 --tag eb3x2 \
  --enc efficientnet_b3.ra2_in1k --scale 2 --crop 384 --bs 12
```

该命令没有覆盖当前 `iters=6000,lr=.001` 默认；不把 fold4 的 crop384/batch12 推广成全部折的历史实参。下面显式选用这组设置作为两家族训练示例，**不是已证实的全部历史命令**：

```bash
for fold in 0 1 2 3 4; do
  python3 train2.py --seed 0 --folds 5 --only_fold "$fold" --tag r34x2 \
    --enc resnet34.a1_in1k --scale 2 --iters 6000 --crop 384 --bs 12 --lr 0.001
  python3 train2.py --seed 0 --folds 5 --only_fold "$fold" --tag eb3x2 \
    --enc efficientnet_b3.ra2_in1k --scale 2 --iters 6000 --crop 384 --bs 12 --lr 0.001
done
```

原训练仅保存最终模型 state_dict，不含优化器状态；无完整命令/环境/确定性保证，不能声称用这些示例能逐字节重训复现上传。当前默认 `train2.py` 为 scale1/crop256/bs16，直接省略参数不等于本 v4 的已知配置。
