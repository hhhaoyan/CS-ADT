import os
import sys
import torch
import json
import tomlkit
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm

# 获取项目的根目录路径，方便后续引用相对路径的文件
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"Project root directory: {project_root}")
# 将项目根目录路径添加到系统路径中，以便可以导入项目中的模块
sys.path.insert(0, project_root)

# 导入 CASTransformer 模块，包括数据处理、评估和模型定义部分
from CASTransformer.data import KTData
from CASTransformer.eval import Evaluator
from CASTransformer.model import CASTransformer  # 导入 CASTransformer 模型
from CASTransformer.visualize import trace_map, heat_map  # 导入可视化模块

# 定义数据目录
DATA_DIR = os.path.join(project_root, "data")
print(f"DATA_DIR: {DATA_DIR}")

# 配置参数解析器，用于从命令行获取运行参数
parser = ArgumentParser()

# 通用选项
parser.add_argument("--device", help="device to run network on", default="cpu")  # 指定设备（CPU或GPU）
parser.add_argument("-bs", "--batch_size", help="batch size", default=64, type=int)  # 批次大小

# 数据集设置，从 toml 文件中加载所有数据集的配置信息
datasets = tomlkit.load(open(os.path.join(DATA_DIR, "datasets.toml")))

# 选择数据集的参数
datasets_help = "choose from a dataset defined in datasets.toml"
parser.add_argument(
    "-d",
    "--dataset",
    help=datasets_help,
    choices=datasets.keys(),
    required=True,
)
# 是否使用问题的ID
parser.add_argument("-p", "--with_pid", help="provide model with pid", action="store_true")

# 模型设置
parser.add_argument("-m", "--model", help="choose model", default="CASTransformer")
parser.add_argument("--d_model", help="model hidden size", type=int, default=128)  # 模型隐藏层大小
parser.add_argument("--n_layers", help="number of layers", type=int, default=1)  # Transformer 层数
parser.add_argument("--n_heads", help="number of heads", type=int, default=8)  # 多头注意力机制的头数
parser.add_argument("--lambda_cl", help="contrastive loss lambda", type=float, default=0.1)  # 对比损失的权重系数
parser.add_argument("--use_d_correct", help="use d_correct in model", action="store_true")  # 是否使用 d_correct 特征
parser.add_argument("--use_d_skill_correct", help="use d_skill_correct in model", action="store_true")  # 是否使用 d_skill_correct 特征

# 测试设置，用于选择已经训练好的模型权重文件
parser.add_argument("-f", "--from_file", help="test existing model file", required=True)
parser.add_argument("-N", help="T+N prediction window size", type=int, default=1)  # 预测窗口大小
parser.add_argument("--seq_id", help="序列索引以进行可视化", type=int, default=0)  # 选择可视化的序列索引


def prepare_tensor(data, dtype, device):
    """
    将数据转换为张量并放置到指定设备上。

    Args:
        data: 要转换的数据。
        dtype: 转换后的数据类型。
        device: 数据放置的设备（CPU或GPU）。

    Returns:
        张量数据，放置到指定设备上。
    """
    if data is not None:
        return torch.as_tensor(data, dtype=dtype).to(device)
    return None


def load_model(args, dataset):
    """
    加载 CASTransformer 模型及其权重。

    Args:
        args: 命令行参数。
        dataset: 数据集的配置信息。

    Returns:
        已加载并设置好评估模式的模型。
    """
    model = CASTransformer(
        n_questions=dataset["n_questions"],  # 问题的数量
        n_pid=dataset["n_pid"] if args.with_pid else 0,  # 如果选择使用 PID 则获取对应的数量
        d_model=args.d_model,  # 模型的隐藏层大小
        n_heads=args.n_heads,  # 多头注意力机制的头数
        n_layers=args.n_layers,  # Transformer 层数
        lambda_fcl=args.lambda_cl,
        lambda_bcl=args.lambda_cl,
        use_d_correct=args.use_d_correct,  # 是否使用 d_correct 特征
        use_d_skill_correct=args.use_d_skill_correct,  # 是否使用 d_skill_correct 特征
    )
    try:
        # 加载指定路径的模型权重
        model.load_state_dict(torch.load(args.from_file, map_location=args.device), strict=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    model.to(args.device)  # 将模型移动到指定设备上
    model.eval()  # 设置模型为评估模式
    return model


def evaluate_batch(batch, model, evaluator, args):
    """
    对单个批次的数据进行模型预测与评估。

    Args:
        batch: 批次数据。
        model: 训练好的 CASTransformer 模型。
        evaluator: 用于评估结果的 Evaluator 对象。
        args: 命令行参数。

    Returns:
        包含预测结果和对应真实标签的数据字典。
    """
    # 定义要提取的字段
    fields = ["q", "s"]  # 问题序列和答案序列
    if "pid" in batch.fields:
        fields.append("pid")
    if "d_correct" in batch.fields:
        fields.append("d_correct")
    if "d_skill_correct" in batch.fields:
        fields.append("d_skill_correct")

    # 从批次数据中获取这些字段
    batch_data = batch.get(*fields)

    # 解包字段
    q, s = batch_data[:2]
    pid = batch_data[2] if "pid" in fields else None
    d_correct = batch_data[3] if "d_correct" in fields else None
    d_skill_correct = batch_data[4] if "d_skill_correct" in fields else None

    # 将字段转换为张量并放置到指定设备上
    q = prepare_tensor(q, torch.long, args.device)
    s = prepare_tensor(s, torch.long, args.device)
    pid = prepare_tensor(pid, torch.long, args.device)
    d_correct = prepare_tensor(d_correct, torch.float, args.device)
    d_skill_correct = prepare_tensor(d_skill_correct, torch.float, args.device)

    try:
        # 使用模型进行预测
        y, *_ = model(q, s, pid, d_correct=d_correct, d_skill_correct=d_skill_correct)  # 调用模型的预测方法
    except Exception as e:
        print(f"Error processing batch: {e}")
        return None

    # 检查切片后的答案是否为空
    if s[:, (args.N - 1):].numel() == 0:
        return None

    # 使用 Evaluator 对预测结果进行评估
    evaluator.evaluate(s[:, (args.N - 1):], torch.sigmoid(y))
    return {
        "y_pred": y.cpu().numpy().tolist(),
        "s_true": s.cpu().numpy().tolist(),
        "pid": pid.cpu().numpy().tolist() if pid is not None else None,
        "d_correct": d_correct.cpu().numpy().tolist() if d_correct is not None else None,
        "d_skill_correct": d_skill_correct.cpu().numpy().tolist() if d_skill_correct is not None else None
    }


def save_results(predictions, evaluator, args, results_dir):
    """
    保存评估指标和预测结果到指定目录。

    Args:
        predictions: 模型的预测结果。
        evaluator: 评估对象。
        args: 命令行参数。
        results_dir: 保存结果的目录。
    """
    try:
        # 保存评估指标
        output_metrics_path = os.path.join(results_dir, f"metrics_N{args.N}.json")
        metrics_report = evaluator.report()  # 获取评估报告
        metrics_output = {
            "args": vars(args),  # 将参数字典化以保存
            "metrics": metrics_report  # 评估指标
        }
        with open(output_metrics_path, "w") as f:
            json.dump(metrics_output, f, indent=2)  # 保存评估指标到文件
        print(f"Metrics saved to {output_metrics_path}")

        # 保存预测结果
        output_predictions_path = os.path.join(results_dir, f"predictions_N{args.N}.json")
        with open(output_predictions_path, "w") as f:
            json.dump(predictions, f, indent=2)  # 保存预测结果到文件
        print(f"Predictions saved to {output_predictions_path}.")
    except Exception as e:
        print(f"Error saving results: {e}")


def visualize_attention(args, model, test_data):
    """
    可视化模型的注意力热力图，显示问题和知识之间的注意力分布。

    Args:
        args: 命令行参数。
        model: 训练好的 CASTransformer 模型。
        test_data: 测试数据集。
    """
    # 获取指定序列的数据
    data = test_data[args.seq_id]
    q = prepare_tensor(data.get("q"), torch.long, args.device)
    s = prepare_tensor(data.get("s"), torch.long, args.device)
    pid = prepare_tensor(data.get("pid"), torch.long, args.device) if args.with_pid else None
    d_correct = prepare_tensor(data.get("d_correct"), torch.float, args.device)
    d_skill_correct = prepare_tensor(data.get("d_skill_correct"), torch.float, args.device)

    # 使用模型进行预测并获取注意力分数
    with torch.no_grad():
        _, _, _, _, (q_scores, k_scores) = model(q, s, pid, d_correct=d_correct, d_skill_correct=d_skill_correct)

    # 定义绘图的序列长度
    plot_seq_len = min(q_scores.size(-1), 12)

    # -----------------------------
    # 问题注意力热力图
    heads = [0, 1, 2, 3]  # 选择要绘制的注意力头
    fig1, ax1 = plt.subplots(1, len(heads), figsize=(4 * len(heads), 4))
    for i, head in enumerate(heads):
        if head >= q_scores.size(1):
            print(f"Head {head} 超出 q_scores 的范围 (n_heads={q_scores.size(1)})")
            ax1[i].axis('off')  # 如果头超出范围，隐藏该子图
            continue
        # 提取特定头和序列长度的注意力分数
        attn = q_scores[0, head, :plot_seq_len, :plot_seq_len].cpu().numpy()
        im = heat_map(ax1[i], attn, cmap="viridis")  # 使用热力图进行可视化
        ax1[i].set_title(f'Head {head}')
    # 为所有子图添加一个共享的色条
    fig1.colorbar(im, ax=ax1.ravel().tolist(), shrink=0.6)
    fig1.suptitle("问题注意力热力图")

    if len(k_scores.shape) == 5:
        num_steps = k_scores.size(2)
        steps = [10, 20, 30, 40]  # 选择一些步长进行绘图
        # 将步长限制在可用范围内
        steps = [step for step in steps if step < num_steps]
    else:
        # 如果 k_scores 形状不同，请根据实际情况调整
        steps = []

    head = 0  # 使用第一个头
    fig2, ax2 = plt.subplots(len(steps), 1, figsize=(8, 2.5 * len(steps)))
    if len(steps) == 1:
        ax2 = [ax2]  # 使其可迭代
    for i, step in enumerate(steps):
        if head >= k_scores.size(1):
            print(f"Head {head} 超出 k_scores 的范围 (n_heads={k_scores.size(1)})")
            ax2[i].axis('off')  # 如果头超出范围，隐藏该子图
            continue
        # 提取特定步长的注意力分数
        attn_k = k_scores[0, head, step, :plot_seq_len, :plot_seq_len].cpu().numpy()
        im = heat_map(ax2[i], attn_k, cmap="plasma")  # 使用热力图进行可视化
        ax2[i].set_title(f'知识头 {head} 在步长 {step}')
    if steps:
        fig2.colorbar(im, ax=ax2, shrink=0.6)
    fig2.suptitle("知识注意力热力图")

    plt.tight_layout()
    plt.show()
    # 可选：保存图像到文件
    # fig1.savefig('question_attention_heatmaps.pdf', bbox_inches='tight')
    # fig2.savefig('knowledge_attention_heatmaps.pdf', bbox_inches='tight')


def main(args):
    """
    主函数，用于加载数据集、模型并进行测试和可视化。

    Args:
        args: 命令行参数。
    """
    # 加载数据集配置
    dataset = datasets[args.dataset]
    seq_len = None  # 强制将 seq_len 设置为 None 以避免数据分块
    test_data = KTData(
        os.path.join(DATA_DIR, dataset["test"]),
        dataset["inputs"],
        seq_len=seq_len,
        batch_size=args.batch_size,
    )

    # 初始化模型
    model = load_model(args, dataset)

    # 创建评估器对象
    evaluator = Evaluator()

    # 准备保存预测结果的目录
    test_dataset_path = os.path.abspath(os.path.join(DATA_DIR, dataset["test"]))
    test_dataset_dir = os.path.dirname(test_dataset_path)
    results_dir = os.path.join(test_dataset_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    predictions = []

    # 测试模型
    with torch.no_grad():
        for batch in tqdm(test_data, desc="Testing"):
            prediction = evaluate_batch(batch, model, evaluator, args)
            if prediction is not None:
                predictions.append(prediction)

    # 保存测试结果
    save_results(predictions, evaluator, args, results_dir)

    # 可视化模型注意力
    visualize_attention(args, model, test_data)


if __name__ == "__main__":
    # 解析命令行参数并启动测试和可视化过程
    args = parser.parse_args()
    print(f"Starting test with arguments: {args}")
    main(args)
