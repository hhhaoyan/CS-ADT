import os
import json
import sys
from argparse import ArgumentParser
import torch.utils.checkpoint as checkpoint
import sys
sys.path.append("D:/CASTransformer")
import CASTransformer

import tomlkit
import torch
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler
import gc
# 其他导入保持不变


# 设置 PYTORCH_CUDA_ALLOC_CONF 环境变量，避免显存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.cuda.empty_cache()

# 解决可能的库冲突问题
os.environ['kmp_DUPLICATE_LIB_OK'] = 'true'

# 获取当前脚本所在的目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目的根目录（假设与 scripts 文件夹同级）
BASE_DIR = os.path.dirname(SCRIPT_DIR)
# 将项目根目录添加到 sys.path
sys.path.insert(0, BASE_DIR)

# 导入自定义模块
from CASTransformer.model import CASTransformer
from CASTransformer.data import KTData
from CASTransformer.eval import Evaluator

DATA_DIR = os.path.join(BASE_DIR, "data")

# 配置主解析器
parser = ArgumentParser()

# 一般选项
parser.add_argument("--device", help="device to run network on", default="cpu")
parser.add_argument("-bs", "--batch_size", help="batch size", default=64, type=int)
parser.add_argument("-tbs", "--test_batch_size", help="test batch size", default=128, type=int)

# 加载数据集配置
with open(os.path.join(DATA_DIR, "datasets.toml"), 'r', encoding='utf-8') as f:
    datasets = tomlkit.load(f)
print(f"Loaded datasets: {datasets}")

# 选择数据集，要求必须选择一个数据集
parser.add_argument(
    "-d",
    "--dataset",
    help="choose from a dataset",
    choices=datasets.keys(),
    required=True,
)
parser.add_argument(
    "-p", "--with_pid", help="provide model with pid", action="store_true"
)
parser.add_argument(
    "--accumulation_steps",
    help="number of gradient accumulation steps",
    type=int,
    default=8
)

# 模型设置
parser.add_argument("-m", "--model", help="choose model", choices=["AKT","DKT","DKVMN","DTransformer","CASTransformer"], required=True)
parser.add_argument("--d_model", help="model hidden size", type=int, default=128)
parser.add_argument("--d_fc", help="dimension of fully connected layer", type=int, default=256)
parser.add_argument("--n_layers", help="number of layers", type=int, default=8)
parser.add_argument("--n_heads", help="number of heads", type=int, default=8)
parser.add_argument("--dropout", help="dropout rate", type=float, default=0.1)
parser.add_argument("--proj", help="projection layer before CL", action="store_true")
parser.add_argument("--hard_neg", help="use hard negative samples in CL", action="store_true")
parser.add_argument("--shortcut", help="use shortcut in the model", action="store_true")
parser.add_argument("--use_d_correct", help="use d_correct as input", action="store_true")
parser.add_argument("--use_d_skill_correct", help="use d_skill_correct as input", action="store_true")
parser.add_argument("--window", help="prediction window size", type=int, default=1)
parser.add_argument('--n_know', type=int, help='use n_know as input', default=16)
# parser.add_argument("--increase_rate", help="emotional factor increase rate", type=float, default=1.01)
# parser.add_argument("--decrease_rate", help="emotional factor decrease rate", type=float, default=0.09)

# 训练设置
parser.add_argument("-n", "--n_epochs", help="training epochs", type=int, default=100)
parser.add_argument(
    "-lr", "--learning_rate", help="learning rate", type=float, default=1e-3
)
parser.add_argument("-l2", help="L2 regularization", type=float, default=1e-5)
parser.add_argument(
    "-cl", "--cl_loss", help="use contrastive learning loss", action="store_true"
)
# 快照设置
parser.add_argument("-o", "--output_dir", help="directory to save model files and logs", required=True)
parser.add_argument(
    "-f", "--from_file", help="resume training from existing model file", default=None
)

# 训练逻辑
def main(args):
    # 准备数据集
    dataset = datasets[args.dataset]
    print(f"Dataset configuration: {dataset}")
    seq_len = dataset.get("seq_len", None)
    n_questions = dataset["n_questions"]
    if n_questions <= 0:
        raise ValueError(f"Warning '{args.dataset}' has invalid n_questions: {n_questions}")

    train_data = KTData(
        os.path.join(DATA_DIR, dataset["train"]),
        inputs=dataset["inputs"],
        seq_len=seq_len,
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_data = KTData(
        os.path.join(
            DATA_DIR, dataset.get("valid", dataset["test"])
        ),
        inputs=dataset["inputs"],
        seq_len=seq_len,
        batch_size=args.test_batch_size,
    )

    args.use_d_correct = "d_correct" in dataset["inputs"]
    args.use_d_skill_correct = "d_skill_correct" in dataset["inputs"]
    args.with_pid = "pid" in dataset["inputs"]

    print(f"Dataset '{args.dataset}' selected.")
    print(f"Using PID: {args.with_pid}")
    print(f"Using d_correct: {args.use_d_correct}")
    print(f"Using d_skill_correct: {args.use_d_skill_correct}")

    if args.output_dir:
        dataset_output_dir = os.path.join(args.output_dir, args.dataset)
        os.makedirs(dataset_output_dir, exist_ok=True)
        config_path = os.path.join(dataset_output_dir, "config.json")
        with open(config_path, "w", encoding='utf-8') as f:
            json.dump(vars(args), f, indent=2)
        print(f"Configuration saved to {config_path}")
    else:
        dataset_output_dir = args.output_dir

    if args.model == "DKT":
        from baselines.DKT import DKT
        model = DKT(dataset["n_questions"], args.d_model)

    elif args.model == "DKVMN":
        from baselines.DKVMN import DKVMN

        model = DKVMN(dataset["n_questions"], args.batch_size)

    elif args.model == "AKT":
        from baselines.AKT import AKT

        model = AKT(
            dataset["n_questions"],
            dataset["n_pid"],
            d_model=args.d_model,
            n_heads=args.n_heads,
            dropout=args.dropout,
        )

    elif args.model == "DTransformer":
        from baselines.DTransformer import DTransformer

        model = DTransformer(
            dataset["n_questions"],
            dataset["n_pid"],
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_know=args.n_know,
            lambda_cl=args.lambda_cl,
            dropout=args.dropout,
            proj=args.proj,
            hard_neg=args.hard_neg,
            window=args.window,
        )

    # 准备模型和优化器
    else:
        from CASTransformer.model import CASTransformer

        n_pid = dataset.get("n_pid", 0) if args.with_pid else 0
        model = CASTransformer(
            dataset["n_questions"],
            dataset["n_pid"],
            d_model=args.d_model,
            d_fc=args.d_fc,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            lambda_fcl=0.1,  # 特征级对比学习的权重
            lambda_bcl=0.1,  # 批次级对比学习的权重
            proj=args.proj,
            hard_neg=args.hard_neg,
            window=args.window,
            shortcut=args.shortcut,
            use_d_correct=args.use_d_correct,
            use_d_skill_correct=args.use_d_skill_correct,
        )

        print(f"Model initialized with n_questions: {model.n_questions}")

    if args.from_file:
        print(f"Loading model state from {args.from_file}")
        model.load_state_dict(torch.load(args.from_file, map_location=args.device))

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.l2
    )
    model.to(args.device)

    # 初始化混合精度缩放器
    scaler = GradScaler()

    # 训练过程
    best = {"auc": 0}
    best_epoch = 0

    for epoch in range(1, args.n_epochs + 1):
        print(f"Start epoch {epoch}")
        model.train()
        it = tqdm(iter(train_data), desc=f"Epoch {epoch}/{args.n_epochs}")

        total_loss = 0.0
        total_pred_loss = 0.0
        total_cl_loss = 0.0
        total_cnt = 0

        optim.zero_grad()

        for i, batch in enumerate(it):
            inputs = batch.get(*dataset["inputs"])
            input_dict = dict(zip(dataset["inputs"], inputs))
            q = input_dict["q"].to(args.device)
            s = input_dict["s"].to(args.device)
            pid = input_dict.get("pid", None)
            if pid is not None:
                pid = pid.to(args.device)
            d_correct = input_dict.get("d_correct", None)
            if d_correct is not None:
                d_correct = d_correct.to(args.device)
            d_skill_correct = input_dict.get("d_skill_correct", None)
            if d_skill_correct is not None:
                d_skill_correct = d_skill_correct.to(args.device)

            if q.max().item() >= n_questions + 1:
                print(f"Warning: q.max() exceeds n_questions. q.max()={q.max().item()}, n_questions={n_questions}")

            # 训练模型
            with autocast():  # 启用混合精度
                if args.model == 'DKVMN':
                    loss = model.get_loss(q, s, d_correct=d_correct, d_skill_correct=d_skill_correct)
                elif args.model == 'CASTransformer':
                    if args.cl_loss:
                        loss, pred_loss, cl_loss = model.get_cl_loss(q, s, pid, d_correct=d_correct,d_skill_correct=d_skill_correct)
                        total_pred_loss += pred_loss.item()
                        total_cl_loss += cl_loss.item()
                        loss = model.get_loss(q, s, pid, d_correct=d_correct, d_skill_correct=d_skill_correct)
                    loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % args.accumulation_steps == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

            total_loss += loss.item() * args.accumulation_steps
            total_cnt += 1

            postfix = {"loss": total_loss / total_cnt}
            if args.cl_loss:
                postfix["pred_loss"] = total_pred_loss / total_cnt
                postfix["cl_loss"] = total_cl_loss / total_cnt
            it.set_postfix(postfix)

            del q, s, pid, d_correct, d_skill_correct
            if 'pred_loss' in locals():
                del loss, pred_loss, cl_loss
            gc.collect()
            torch.cuda.empty_cache()

        if args.accumulation_steps > 1 and (i + 1) % args.accumulation_steps != 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

        model.eval()
        evaluator = Evaluator()

        with torch.no_grad():
            it = tqdm(iter(valid_data), desc=f"Validation Epoch {epoch}")
            for batch in it:
                inputs = batch.get(*dataset["inputs"])
                input_dict = dict(zip(dataset["inputs"], inputs))
                q = input_dict["q"].to(args.device)
                s = input_dict["s"].to(args.device)
                pid = input_dict.get("pid", None)
                if pid is not None:
                    pid = pid.to(args.device)
                d_correct = input_dict.get("d_correct", None)
                if d_correct is not None:
                    d_correct = d_correct.to(args.device)
                d_skill_correct = input_dict.get("d_skill_correct", None)
                if d_skill_correct is not None:
                    d_skill_correct = d_skill_correct.to(args.device)

                with autocast():
                    y, *_ = model.predict(q, s, pid=pid, d_correct=d_correct, d_skill_correct=d_skill_correct)
                    evaluator.evaluate(s, torch.sigmoid(y))

        r = evaluator.report()
        print(f"Validation results after epoch {epoch}: {r}")

        if r["auc"] > best["auc"]:
            best = r
            best_epoch = epoch
            if dataset_output_dir:
                model_path = os.path.join(
                    dataset_output_dir, f"model-{epoch:03d}-{r['auc']:.4f}.pt"
                )
                print(f"Saving best model to: {model_path}")
                torch.save(model.state_dict(), model_path)

                best_result_path = os.path.join(dataset_output_dir, "best_results.json")
                best_results = {
                    "best_epoch": best_epoch,
                    "best_auc": best["auc"],
                    "best_results": {k: f"{v:.4f}" for k, v in best.items()},
                }
                with open(best_result_path, "w", encoding="utf-8") as f:
                    json.dump(best_results, f, indent=2)
                print(f"Best results saved to: {best_result_path}")

    formatted_best = {k: f"{v:.4f}" for k, v in best.items()}

    return best_epoch, best

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Training configuration: {args}")
    best_epoch, best = main(args)
    print(f"Best epoch: {best_epoch}")
    print(f"Best result: {{ {', '.join([f'{k}: {v:.4f}' for k, v in best.items()])} }}")
