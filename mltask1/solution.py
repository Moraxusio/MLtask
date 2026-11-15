#!/usr/bin/env python3
"""机器学习课程设计 任务一：植物幼苗图像分类（5 类）

跑一次这个脚本就能出结果，中间不用手动操作：
读数据 -> 训练 -> 交叉验证看准确率 -> 预测测试集 -> 存成提交用的 csv

数据长这样：
    dataset-for-task2/dataset-for-task2/
        train/<类别名>/*.png     每个类别 100 张，一共 500 张
        test/*.png               378 张，要我们预测的

思路：
    训练集只有 500 张，从零开始训肯定不够，所以用迁移学习——
    拿 ImageNet 上预训练好的 ConvNeXt-Tiny，把最后的分类层换成 5 类的，
    然后在我们的数据上微调。
    因为数据少，划一个固定验证集太浪费（而且结果看运气），
    所以用 5 折交叉验证：每次拿 400 张训练、100 张验证，轮 5 次。
    最后 5 个模型对测试集的预测概率取平均，再取最大的那个类别。

怎么跑：
    python solution.py  # 最终 ConvNeXt-Tiny@384、EMA=0.99、noMixup 配置
    # 数据路径及完整配置命令见 README.md；运行会实际训练并预测。

有几个地方是我试出来的，写在这里免得忘了：
  1. 一开始我是「哪个 epoch 验证集准确率最高就存哪个」，后来发现这样不太对——
     相当于拿验证集挑了一次参数，再用同一个验证集报成绩，分数会偏高。
     改成了 EMA（权重滑动平均），不用挑 epoch，分数更实在。
  2. 不存模型文件。每折训完马上预测测试集、把概率加起来，然后模型就扔掉，
     省硬盘（本来 5 个模型要占 1 个 G）。
  3. 要不要用 TTA 我没有自己定，让程序把用和不用的准确率都算出来，哪个高用哪个。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ---------------------------------------------------------------- 一些常量

# 类别名字写死在这里，顺序千万不能改。
# 因为标签是按这个列表的下标编号的（Black-grass=0, Common wheat=1 ...），
# 训练、预测、写 csv 三个地方都用它，顺序一变就全乱了。
CLASSES = [
    "Black-grass",
    "Common wheat",
    "Loose Silky-bent",
    "Scentless Mayweed",
    "Sugar beet",
]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# 这两组数是 ImageNet 数据集的均值和标准差，用预训练模型就得按它来归一化，
# 不然输入分布和模型当初训练时不一样，效果会差
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# TTA（测试时增强）：预测的时候把图片翻转几下，分别预测再把概率平均一下，
# 相当于让模型多看几眼，一般能稳一点。
# 这里敢用翻转和转 180 度，是因为这些照片都是从上往下拍的，
# 把图翻过来它还是同一种草，标签不会变。
TTA_OPS = OrderedDict(
    [
        ("none", lambda x: x),
        ("hflip", lambda x: torch.flip(x, dims=[3])),
        ("vflip", lambda x: torch.flip(x, dims=[2])),
        ("rot180", lambda x: torch.flip(x, dims=[2, 3])),
    ]
)


# ---------------------------------------------------------------- 读数据

def list_train_samples(root: Path) -> tuple[list[Path], np.ndarray]:
    """把训练集所有图片的路径和对应标签找出来。

    返回 (路径列表, 标签数组)。
    """
    paths: list[Path] = []
    labels: list[int] = []
    for cls in CLASSES:
        cls_dir = root / "train" / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"缺少类别目录: {cls_dir}")
        # 这里加 sorted 是为了每次运行读到的顺序都一样，
        # 不然交叉验证切出来的 5 份每次都不同，没法复现结果
        for p in sorted(cls_dir.glob("*.png")):
            paths.append(p)
            labels.append(CLASS_TO_IDX[cls])
    if not paths:
        raise RuntimeError(f"{root / 'train'} 下没有找到任何 png")
    return paths, np.array(labels)


def list_test_samples(root: Path) -> list[Path]:
    """把测试集图片路径找出来，同样排好序。"""
    paths = sorted((root / "test").glob("*.png"))
    if not paths:
        raise RuntimeError(f"{root / 'test'} 下没有找到任何 png")
    return paths


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    """图片预处理。train=True 的时候做数据增强，验证和测试的时候不做。

    训练集就 500 张，模型很容易把这些图背下来（过拟合），
    所以增强加得比较多：随机裁、翻转、旋转、改颜色，最后再随机擦掉一小块。
    每个 epoch 看到的图都不太一样，模型就没那么容易死记硬背了。
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.85, 1.18)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
            ]
        )
    # 验证和测试不能随机，不然每次跑出来的分数都不一样。
    # 先放大到 1.14 倍再从中间裁，这样图里草的大小和训练时差不多
    # （训练用的 RandomResizedCrop 平均下来也是裁掉一圈）
    return transforms.Compose(
        [
            transforms.Resize(int(round(img_size * 1.14))),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class SeedlingDataset(Dataset):
    """自定义的 Dataset，按路径列表一张张读图。

    测试集没有标签，这时候 labels 传 None 就行。
    """

    def __init__(self, paths: list[Path], labels: np.ndarray | None, transform) -> None:
        if labels is not None and len(paths) != len(labels):
            raise ValueError("paths 与 labels 长度不一致")
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        # 有些 png 是 4 通道的（带透明度），不转成 RGB 的话后面模型会报错
        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.transform(img)
        if self.labels is None:
            # 测试集没标签，但 DataLoader 要求返回的格式统一，
            # 所以随便返回个 0 凑数，预测的时候根本不会用到它
            return x, 0
        return x, int(self.labels[idx])


# ---------------------------------------------------------------- 搭模型

def create_model(arch: str) -> nn.Module:
    """建模型：下载 ImageNet 预训练好的模型，把最后一层换成 5 类输出。

    前面那些卷积层学到的边缘、纹理这些特征是通用的，可以直接拿来用，
    只有最后分类的那一层要换掉重新学（原来是 1000 类，我们只有 5 类）。

    注：权重是从 torchvision 官方地址下的。
    我本来想用 timm 库（模型更多），但它要连 HuggingFace，这边连不上，所以换了。
    """
    if arch == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"不支持的 arch: {arch}")
    return m


def param_groups(model: nn.Module, arch: str, head_lr: float, backbone_lr: float) -> list[dict]:
    """给模型的两部分设不同的学习率。

    最后那个分类层是新加的、参数还是随机的，得用大一点的学习率快点学；
    前面的卷积层已经在 ImageNet 上训得很好了，学习率必须调小，
    不然几步就把人家辛苦学到的特征给改坏了（我一开始统一用 5e-4，掉了好几个点）。
    这里头的学习率是骨干的 10 倍。
    """
    if arch != "convnext_tiny":
        raise ValueError(f"不支持的 arch: {arch}")
    head_prefix = "classifier.2."
    head, backbone = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (head if name.startswith(head_prefix) else backbone).append(p)
    return [{"params": backbone, "lr": backbone_lr}, {"params": head, "lr": head_lr}]


class ModelEMA:
    """权重的滑动平均（EMA）。

    训练的时候模型权重一直在抖，最后一轮的权重不一定是最好的。
    EMA 的做法是另外存一份权重，每步都往当前权重的方向挪一小点：
        新的影子权重 = 0.99 * 旧影子权重 + 0.01 * 当前权重
    相当于把最近很多步的权重平均了一下，比单独某一步要稳。

    加这个主要是为了不用再「挑验证集上最好的那个 epoch」。
    挑 epoch 等于偷看了验证集，报出来的准确率会虚高；
    用 EMA 就不用挑了，直接拿这份平均权重去评估，分数更可信。
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        # 单独建一个一模一样的模型来存平均后的权重。
        # 它只是用来存数的，不参与反向传播，所以 requires_grad 全关掉
        self.module = create_model_like(model)
        self.module.load_state_dict(model.state_dict())
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """每训练一个 batch 就调一次，把影子权重往当前权重挪一点。"""
        d = self.decay
        for ema_v, cur_v in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                # 浮点的权重才做平均
                ema_v.mul_(d).add_(cur_v.detach(), alpha=1.0 - d)
            else:
                # 有些是整数类型的（比如 BN 记录「已经处理了多少个 batch」的计数器），
                # 整数做平均没有意义，直接复制过来
                ema_v.copy_(cur_v)


def create_model_like(model: nn.Module) -> nn.Module:
    """照着传进来的模型，再建一个结构相同的新模型（给 EMA 存权重用）。

    用的是前面挂在模型上的 _arch_tag，记着这个模型是哪个架构的。
    """
    return create_model(model._arch_tag)


# ---------------------------------------------------------------- 训练部分

def set_seed(seed: int) -> None:
    """固定随机种子，这样每次跑的结果都一样，方便对比不同参数的效果。

    四个库的种子都要设，漏了哪个都可能导致结果对不上。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """mixup 数据增强：把两张图按比例叠在一起，标签也按同样比例混合。

    比如 0.7 的小麦图 + 0.3 的甜菜图，那标签就是 70% 小麦、30% 甜菜。
    这是一种防过拟合的手段。

    返回 (叠加后的图, 第一个标签, 第二个标签, 混合比例 lam)，
    算损失的时候两个标签的交叉熵按 lam 加权相加。

    注：我试过了，这个数据集上没什么用，所以默认是关掉的（见下面参数说明）。
    """
    lam = float(np.random.beta(alpha, alpha))
    # 强制让 lam 不小于 0.5，保证第一张图始终是「主要的」那张。
    # 不这么做的话 0.3 和 0.7 其实是同一种混合，只是主次颠倒，白白增加随机性
    lam = max(lam, 1.0 - lam)
    # 把这一批图随机打乱当作「第二张图」，省得再读一遍数据
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device, tta: list[str]) -> np.ndarray:
    """用模型预测，返回每张图属于各个类别的概率，形状是 (图片数, 5)。

    tta 里传了几种变换，就把图按这几种方式各变一次分别预测，最后概率取平均。
    只传 ["none"] 就是不做 TTA 的普通预测。
    """
    model.eval()  # 切成评估模式，这样 dropout 之类的就不起作用了
    chunks = []
    use_amp = device.type == "cuda"
    for x, _ in loader:  # 标签用不上，用 _ 接住
        x = x.to(device, non_blocking=True)
        acc = None  # 用来累加几种 TTA 的概率
        for name in tta:
            xa = TTA_OPS[name](x)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                # softmax 把模型输出变成加起来等于 1 的概率
                p = model(xa).float().softmax(1)
            acc = p if acc is None else acc + p
        chunks.append((acc / len(tta)).cpu().numpy())  # 除以个数得到平均概率
    return np.concatenate(chunks, 0)  # 把每个 batch 的结果拼成一个大矩阵


def train_one_fold(
    args,
    tr_paths: list[Path],
    tr_labels: np.ndarray,
    device,
    fold: int,
) -> tuple[nn.Module, nn.Module | None]:
    """训练其中一折。

    返回 (训练完的模型, EMA 模型)。如果没开 EMA，第二个返回 None。
    """
    tf_train = build_transforms(args.img_size, train=True)
    tr_ds = SeedlingDataset(tr_paths, tr_labels, tf_train)
    tr_ld = DataLoader(
        tr_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False,
        persistent_workers=args.workers > 0,
    )

    model = create_model(args.arch)
    model._arch_tag = args.arch  # 记一下用的哪个架构，EMA 建影子模型时要照着建
    model = model.to(device)  # 搬到 GPU 上
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        ema.module = ema.module.to(device)

    # 损失函数用交叉熵，分类任务的标配。
    # label_smoothing 是标签平滑：本来正确类别的目标是 1、其他是 0，
    # 5 类且 label_smoothing=0.1 时，PyTorch 均匀平滑后真类为 0.92、其余各 0.02
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    # 优化器用 AdamW，比 SGD 省心，不用费劲调学习率
    optimizer = torch.optim.AdamW(
        param_groups(model, args.arch, args.head_lr, args.backbone_lr),
        weight_decay=args.weight_decay,
    )
    # 学习率调度：前几轮从小慢慢升上去（warmup），之后按余弦曲线慢慢降到 0。
    # warmup 是因为刚开始分类层参数是随机的，梯度很乱，
    # 一上来就用大学习率容易把预训练权重带偏；后面降学习率是为了收敛得稳一些
    warmup = min(3, max(1, args.epochs // 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda e: (e + 1) / warmup if e < warmup
        else 0.5 * (1.0 + np.cos(np.pi * (e - warmup) / max(1, args.epochs - warmup))),
    )

    use_amp = device.type == "cuda"
    for ep in range(args.epochs):
        model.train()  # 切回训练模式
        tot_loss, n_correct, n = 0.0, 0, 0  # 统计这一轮的损失和正确数
        for x, y in tr_ld:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # autocast 是混合精度训练，用 bf16 算得更快也更省显存。
            # 这里不需要配 GradScaler（网上很多教程会加），
            # 那个是给 fp16 用的，因为 fp16 范围小容易梯度下溢；
            # bf16 的范围和 fp32 一样大，不会有这个问题
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                if args.mixup_alpha > 0:
                    xm, ya, yb, lam = mixup_batch(x, y, args.mixup_alpha)
                    logits = model(xm)
                    loss = lam * criterion(logits, ya) + (1.0 - lam) * criterion(logits, yb)
                else:
                    logits = model(x)
                    loss = criterion(logits, y)
            # 反向传播三件套：清梯度 -> 算梯度 -> 更新参数
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # 梯度裁剪，防止偶尔出现特别大的梯度把参数冲飞
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)  # 每个 batch 都更新一次影子权重
            bs = y.size(0)
            tot_loss += loss.item() * bs  # 乘 batch 大小，最后除总数才是真平均
            # 这个训练准确率只能当个参考看，因为开了 mixup 的话标签是混合的，
            # 拿混合前的 y 去比对本身就不严谨。真正的成绩要看验证集
            n_correct += (logits.argmax(1) == y).sum().item()
            n += bs
        scheduler.step()  # 一个 epoch 结束，调整学习率
        if (ep + 1) % max(1, args.log_every) == 0 or ep == args.epochs - 1:
            print(
                f"  [fold {fold}] ep {ep + 1:02d}/{args.epochs} "
                f"loss {tot_loss / n:.4f}  train_acc(参考) {n_correct / n:.3f}",
                flush=True,
            )
    return model, (ema.module if ema is not None else None)


# ---------------------------------------------------------------- 主程序

def main() -> None:
    # 所有参数都做成命令行可调的，调参的时候不用改代码
    ap = argparse.ArgumentParser(description="植物幼苗 5 分类，单文件端到端方案")
    ap.add_argument("--data-root", default="dataset-for-task2/dataset-for-task2")
    ap.add_argument("--out", default="submission-for-task2.csv")
    ap.add_argument("--arch", default="convnext_tiny", choices=["convnext_tiny"])
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--backbone-lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    # 保留原实验默认值：mixup 默认关闭；此源码包不声明历史分数。
    ap.add_argument("--mixup-alpha", type=float, default=0.0, help="0 表示关闭 mixup")
    ap.add_argument("--no-mixup", action="store_true", help="等价于 --mixup-alpha 0")
    # EMA 衰减系数沿用原默认值；值越小，越贴近当前权重。
    ap.add_argument("--ema-decay", type=float, default=0.99, help="0 表示关闭 EMA")
    ap.add_argument("--seed", type=int, default=42)
    # 下面这个是读数据用的进程数
    ap.add_argument("--workers", type=int, default=8)
    # 每几个 epoch 打印一次日志，不然屏幕上刷太多
    ap.add_argument("--log-every", type=int, default=5)
    # 评估结果存成 json 的路径
    ap.add_argument("--report", default="cv_report.json")
    args = ap.parse_args()

    if args.no_mixup:
        args.mixup_alpha = 0.0

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data_root)

    paths, labels = list_train_samples(root)
    test_paths = list_test_samples(root)
    print(
        f"训练 {len(paths)} 张 / 测试 {len(test_paths)} 张 / {NUM_CLASSES} 类 | device={device}\n"
        f"配置: {args.arch} @{args.img_size}, {args.epochs} ep x {args.folds} 折, "
        f"mixup={args.mixup_alpha}, ema={args.ema_decay}",
        flush=True,
    )

    tf_eval = build_transforms(args.img_size, train=False)
    test_ld = DataLoader(
        SeedlingDataset(test_paths, None, tf_eval), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )

    # 不用 TTA 和用 4 种 TTA 的结果都记下来，最后比一比哪个准确率高就用哪个，
    # 不靠猜。oof 存验证集的预测，test_sum 把 5 折对测试集的预测概率累加起来
    tta_variants = {"none": ["none"], "tta4": list(TTA_OPS.keys())}
    # OOF = out of fold，每张训练图在「它没参与训练的那一折」上的预测结果。
    # 500 张图攒齐之后就能算出一个整体准确率，比单看某一折靠谱
    oof = {k: np.zeros((len(paths), NUM_CLASSES), np.float32) for k in tta_variants}
    test_sum = {k: np.zeros((len(test_paths), NUM_CLASSES), np.float32) for k in tta_variants}

    # StratifiedKFold 是分层抽样，保证每一折里 5 个类别的比例都一样（各 20 张），
    # 普通 KFold 可能切出某折某个类别特别少的情况
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    t_start = time.time()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(paths, labels)):
        t0 = time.time()
        # 每折用不一样的种子，这样 5 折不是完全重复的实验；
        # 加上 fold 而不是随便取，是为了单独重跑某一折时还能复现
        set_seed(args.seed + fold)

        model, ema_model = train_one_fold(
            args, [paths[i] for i in tr_idx], labels[tr_idx], device, fold
        )
        # 开了 EMA 就用平均后的权重来预测，没开就用训练完的权重。
        # 注意这里没有「挑最好的 epoch」这一步，就是故意的（原因见文件开头说明）
        infer_model = ema_model if ema_model is not None else model

        va_ld = DataLoader(
            SeedlingDataset([paths[i] for i in va_idx], labels[va_idx], tf_eval),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
        msg = []
        for key, tta in tta_variants.items():
            # 预测这一折的验证集，填到 oof 对应位置上
            oof[key][va_idx] = predict_probs(infer_model, va_ld, device, tta)
            # 顺手把测试集也预测一遍，概率累加起来，等 5 折跑完再除以 5
            test_sum[key] += predict_probs(infer_model, test_ld, device, tta)
            # argmax 取概率最大的那个类别作为预测结果
            acc = accuracy_score(labels[va_idx], oof[key][va_idx].argmax(1))
            msg.append(f"{key} {acc:.4f}")
        print(
            f"  [fold {fold}] 完成 | val acc: {'  '.join(msg)} | {time.time() - t0:.0f}s",
            flush=True,
        )

        # 这一折用完了，把模型删掉、清空显存，不然 5 个模型会把显存占满
        del model, ema_model, infer_model
        torch.cuda.empty_cache()

    # ---- 5 折都跑完了，汇总看结果，并决定要不要用 TTA ----
    print(f"\n===== 交叉验证汇总（总耗时 {time.time() - t_start:.0f}s）=====")
    oof_acc = {k: accuracy_score(labels, v.argmax(1)) for k, v in oof.items()}
    for k, a in oof_acc.items():
        print(f"  OOF 准确率 [{k:5s}] = {a:.4f}")
    # 哪种方案准确率高就用哪种，让数据说话
    best_key = max(oof_acc, key=oof_acc.get)
    print(f"  -> 按 OOF 选用: {best_key}")

    pred = oof[best_key].argmax(1)
    print(
        "\n分类报告:\n",
        classification_report(labels, pred, target_names=CLASSES, digits=4, zero_division=0),
    )
    # 混淆矩阵能看出来模型具体把谁认成了谁，
    # 第 i 行第 j 列的数 = 真实是第 i 类、被预测成第 j 类的图片数量，
    # 对角线上的是分对的
    print("混淆矩阵 (行=真实, 列=预测):\n", confusion_matrix(labels, pred))

    # ---- 把测试集的预测结果写成提交用的 csv ----
    # 5 折的概率加起来除以 5 得到平均概率，这就是模型集成，比单个模型稳
    probs = test_sum[best_key] / args.folds
    df = pd.DataFrame(
        {"ID": [p.name for p in test_paths], "Category": [CLASSES[i] for i in probs.argmax(1)]}
    )
    df.to_csv(args.out, index=False)

    # 顺便看一下模型有多"自信"：最大概率越接近 1 说明越确定
    conf = probs.max(1)
    print(f"\n已写出 {args.out}（{len(df)} 行）")
    print("预测类别分布:\n", df["Category"].value_counts().to_string())
    print(f"平均最大概率 = {conf.mean():.4f}，置信度 < 0.5 的样本数 = {int((conf < 0.5).sum())}")

    # 提交前自己检查一遍，这几项哪个不对提交上去都是白交。
    # 用 assert 是因为出错了就该直接报错停下来，不能让错文件悄悄生成
    assert set(df["ID"]) == {p.name for p in test_paths}, "ID 与 test 目录不一致"
    assert df["ID"].is_unique and not df.isna().any().any(), "存在重复 ID 或缺失值"
    assert set(df["Category"]) <= set(CLASSES), "出现非法类别名"
    print("提交文件自检通过: 行数/ID 集合/无重复/无缺失/类别合法")

    # 把这次的参数和结果存成 json，方便写报告的时候查，也免得调参调到后面忘了
    if args.report:
        Path(args.report).write_text(
            json.dumps(
                {
                    "config": vars(args),
                    "oof_acc": {k: float(v) for k, v in oof_acc.items()},
                    "selected": best_key,
                    "confusion_matrix": confusion_matrix(labels, pred).tolist(),
                    "classes": CLASSES,
                },
                indent=2, ensure_ascii=False, default=str,
            ),
            encoding="utf-8",
        )
        print(f"评估明细已存至 {args.report}")


if __name__ == "__main__":
    main()
