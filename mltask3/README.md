# 任务三：实际 t3 提交的四组黄斑中心凹定位流程

## 标准最终版本

本整理包以最终提交文件 **`t3.csv`** 为标准最终版本：它与原项目 `submission_v2_4set.csv` 字节一致，SHA-256 为 `6f7edc5fe2e363d40aa60cdc8abe77f9772f672006f57b1aad1e3016f52148d4`。原项目另一个名为 `submission.csv` 的文件是七组、30 个底层模型的版本，**不是本次最终交付**。此前 README 的 Tiny 两阶段示例也不是实际 t3 流程，现已更正；这是文档/打包范围修正，不是重新选择模型或修改最终提交文件。

实际方案均为 stage 1 的 ConvNeXt-Small：预训练骨干、四尺度融合热图头、Gaussian/KL + 坐标 L1、局部 soft-argmax、水平翻转 TTA。JSON 内是半分辨率像素中心坐标，输出转换为原图坐标 `(p_half + 0.5) * 2 - 0.5`。

| 输入顺序 | 预测组及文件（相对预测根目录） | 输入宽×高 | 底层权重 | 组权重 | 每个底层模型的有效权重 |
|---|---|---|---|---|---|
| 1 | `s1_cns/centers.json` | 1024×704 | `fold0.pt`–`fold4.pt`，5 个 | 1/4 | 1/20 |
| 2 | `s1_cns_r05/centers.json` | 512×352 | `fold0.pt`–`fold4.pt`，5 个 | 1/4 | 1/20 |
| 3 | `full_cns/test_full.json` | 1024×704 | `full0.pt`–`full1.pt`，2 个 | 1/4 | 1/8 |
| 4 | `full_cns_r05/test_full.json` | 512×352 | `full0.pt`–`full2.pt`，3 个 | 1/4 | 1/12 |

共 **15 个底层模型，不是 15 模型等权**。CV JSON 的测试部分已经是五折模型均值，full JSON 分别已是 2/3 个全量模型均值；提交器只再做上述四组等权平均。组内默认两视图翻转 TTA 不另算模型。固定输入顺序保留历史浮点求和顺序，不排序、不拟合最终提交文件。

## 文件、数据与依赖

仅保留实际流程需要的 `src/fovea.py`、`src/make_sub.py`。移除无关 `simple.py`、`yolo_data.py`、`yolo_fovea.py`，不打包提示答案、测试标签、数据或权重。`fovea.py` 进一步仅保留 **stage 1、ConvNeXt-Small、half_cache + none、res=1/0.5、CV 和 full**。已删除 stage 2、HRNet/其他骨干、外部数据采样、Graham、全分辨率缓存/变换、strong 增强、辅助回归损失、EMA 实验、训练中验证及单折运行分支。注意 `--full` 指全量标注数据训练，不是全分辨率图像。

```text
mltask3/
  README.md
  src/{fovea,make_sub}.py
  data/detection/                         # 自备或用环境变量指向外部
    train/0001.jpg ... 0080.jpg
    test/0081.jpg ... 0100.jpg
    fovea_localization_train_GT.csv       # 唯一任务监督，data,Fovea_X,Fovea_Y
    sample_submission.csv                # 仅使用 ImageID，忽略 value
  weights/convnext_small.fb_in22k_ft_in1k.safetensors
  out/                                   # 运行时缓存、权重、预测
  logs/                                  # 运行时日志
```

`ROOT` 由源码位置确定。`MLTASK3_DATA_DIR` 覆盖数据目录，`MLTASK3_WEIGHTS_DIR` 覆盖预训练权重目录；都应传绝对路径，默认使用上述本地布局。训练输出只写当前整理包的 `out/`、`logs/`，不要把整理包放进原只读项目。更换数据时必须使用干净缓存目录/新工作副本，避免沿用旧 `out/*cache.pt`。

- 仅 JSON 提交复现：Python、NumPy、pandas、Pillow；无需 torch/GPU。
- 训练另需 PyTorch、timm、safetensors 和支持 CUDA/BF16 的 GPU；历史依赖未锁定，不能编造完整环境锁文件。
- 本次提交重放环境为 Python 3.10.18、NumPy 1.26.4、pandas 2.3.2、Pillow 11.3.0；这不是历史训练环境证明。

## 四方案训练命令（配置重建，未在本次执行）

在 `mltask3/` 下运行。以下命令从对应原始 `logs/<tag>.log` **首行已解析 CLI JSON** 转写，不是声称找到了原 shell 命令。完整首行见 实验记录。false 布尔选项通过不传开关表达。先自行准备合法数据和预训练权重：

```bash
export MLTASK3_DATA_DIR="/absolute/path/to/data/detection"
export MLTASK3_WEIGHTS_DIR="/absolute/path/to/pretrained/weights"
mkdir -p out logs

python3 src/fovea.py --stage 1 --arch convnext_small \
  --steps 2000 --bs 8 --lr 0.0001 --sigma 8 --l1w 0.5 \
  --folds 5 --seed 1 --tag s1_cns --centers ""

python3 src/fovea.py --stage 1 --arch convnext_small \
  --steps 2000 --bs 16 --lr 0.0001 --sigma 4 --l1w 0.5 \
  --folds 5 --seed 1 --tag s1_cns_r05 --centers "" \
  --prep none --res 0.5 --own_frac 0.5 --ext_until 0.8

python3 src/fovea.py --stage 1 --arch convnext_small \
  --steps 2000 --bs 8 --lr 0.0001 --sigma 8 --l1w 0.5 \
  --folds 5 --seed 8 --tag full_cns --centers "" --full \
  --prep none --res 1.0 --val_every 0 --n_full 2 \
  --regw 0 --only_fold -1 --own_frac 0.5 --ext_until 0.8

python3 src/fovea.py --stage 1 --arch convnext_small \
  --steps 2000 --bs 16 --lr 0.0001 --sigma 4 --l1w 0.5 \
  --folds 5 --seed 7 --tag full_cns_r05 --centers "" --full \
  --prep none --res 0.5 --val_every 0 --n_full 3 \
  --regw 0 --only_fold -1 --own_frac 0.5 --ext_until 0.8
```

为保持以上配置重建命令兼容，无作用 CLI 仅接受：`--centers ""`、`--prep none`、`--val_every 0`、`--regw 0`、`--only_fold -1`、`--own_frac 0.5`、`--ext_until 0.8`。它们不会恢复已删除的实验分支；非上述值会报错。`--stage` 仅接受 1，`--arch` 仅接受 `convnext_small`（也是当前默认），`--res` 仅接受 1/0.5。`--strong`、`--ext`、`--ema`、`--s2*` 已删除。日志的参数字段因此少于旧脚本，不应把日志字段差异当成数学路径变化。

2000 是 optimizer steps，不是 epochs。CV 固定 `rng=123` 排列取模五折，每折64训练/16验证；当前代码模型 seed 为 `seed*100+fold`（100–104），full 为 `seed*100+50+k`（850–851、750–752）。full 分支仅全80张训练并输出测试预测，`--folds 5` 在该分支不产生 OOF。

### 历史代码/配置缺口（不可隐去）

- `s1_cns` 首行只到 `full=false`，没有 `res/prep/strong/ext/regw/ema/...`；不把后来默认值伪装为当时显式记录。其1024×704来自历史证据/代码约定。`s1_cns_r05` 明确 `res=0.5,prep=none,strong=false,ext=false`，但没有 `regw/ema/n_full/val_every/only_fold/s2*`。两组 full 明确 `regw=0,n_full=2/3`，仍未记录 `ema/s2*`。
- 精简前脚本对缺失参数采用当时现存源码的默认值（如 `ema=0,regw=0,prep=none,res=1`）；本次保留这些实际路径，删除未启用的 EMA/外部数据等分支。这仅说明配置重建命令的当前行为，**不是历史配置补证**；不使用外部 IDRiD 缓存，也不发明历史旧源码。
- 本次逐份检查 15 个 checkpoint 的 ZIP `data.pkl` 元数据字符串（不反序列化权重、不执行模型）：两组 CV 的10份均没有 `reg.*` 参数，两个 full 组的5份均含 `reg.0.weight/bias`、`reg.2.weight/bias`。当前 `HeatNet` 无条件创建并前向计算辅助头；`--regw 0` 仅为兼容参数，辅助损失实现已删除，但不删除头。保留 `lat → reg → head` 的构造顺序、reg 两层 Linear/零初始化、优化器参数分组和前向计算，避免改变当前实现的 RNG/模型路径。历史 10 个 CV 模型没有 reg，无法直接 strict load 到当前结构，也**不能用当前脚本精确复训**；后来新增头会改变热图头初始化的随机数序列。
- 因此以上是**四组配置重建训练命令**，不是逐数值复训保证；历史旧版源码/Git、部分配置和依赖版本缺失。本次没有删头或宽松加载来掩盖缺口，没有训练或模型推理。此前已成功执行的是下面的**已有预测 JSON → 提交 CSV**重放；提交器保持固定坐标融合与格式转换流程。分支范围、逐函数 AST/CLI/公式检查及限制见 代码版本说明。

## 精确四组提交命令（真实流程）

完成训练后可使用本地 `out/`；重放现有产物时将 `PRED_ROOT` 指向只读原产物目录。`make_sub.py` 不读取训练 GT 或最终提交文件 `t3.csv`，只读四 JSON、模板 `ImageID` 和测试图像尺寸以校验坐标边界；绝不读取模板 value 或测试答案。

```bash
export MLTASK3_DATA_DIR="/absolute/path/to/data/detection"
PRED_ROOT="/absolute/path/to/out"
OUTPUT="/absolute/path/to/reproduced/t3.csv"
mkdir -p "$(dirname "$OUTPUT")"
python3 -B src/make_sub.py "$OUTPUT" \
  "$PRED_ROOT/s1_cns/centers.json" \
  "$PRED_ROOT/s1_cns_r05/centers.json" \
  "$PRED_ROOT/full_cns/test_full.json" \
  "$PRED_ROOT/full_cns_r05/test_full.json"
```

没有硬编码坐标、手工修补、四舍五入拟合或读取参照 CSV 的回路。外部比较必须在生成之后单独进行；浮点序列化若不同，应分别报告逐行数值与字节差异，不把数值相近称为字节一致。生成的 `t3.csv` 的40行文本、数值及全部1205字节均与最终提交文件 `t3.csv` 和原 `submission_v2_4set.csv` 一致，SHA-256 同上。实际运行、输入哈希和比较结果见 实验记录。

## OOF 与数据局限

仅两个 CV 组可在训练 ID 1–80 上各取对应留出预测，再等权平均并还原原图坐标。重算坐标 RMSE 为 **17.916894642 px**（`sqrt(sum(dx²+dy²)/(2N))`），不是平均欧氏距离。单组分别为18.590256276、17.946874702 px。**完整四组没有 OOF**：两组 full 见过全部80张训练图，不能用它们构造独立 OOF，更不把17.916894642归给最终15模型集成。

训练 `0039.jpg` 与 `0077.jpg` 字节重复，却分到 fold4/fold2，导致两折验证图在训练侧有相同副本；标注也略有差异。这一既有跨折泄漏没有因整理代码而消失。未纳入患者分组、近重复审查及嵌套模型选择，结果只能作为带限制的历史 OOF；训练数据划分和模型配置按照实验方案固定。实验使用课程任务提供的数据和训练标注，未引入外部测试标注。报告仅引用带上述限制的训练OOF，不宣称官方成绩或独立盲测。
