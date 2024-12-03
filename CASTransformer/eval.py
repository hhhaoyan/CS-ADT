import numpy as np
import torch
from sklearn import metrics  # 用于计算各种评估指标

class Evaluator:
    def __init__(self):
        # 储存真实标签
        self.y_true = []
        # 储存预测概率
        self.y_pred = []

    def evaluate(self, y_true, y_pred):
        # 创建掩码，只保留真实标签中 >= 0 的值，过滤掉无效标签。
        mask = y_true >= 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # 如果过滤后没有样本，跳过
        if y_true.size(0) == 0:
            print("Evaluator: No valid samples after masking.")
            return

        # 将张量转换为 CPU，并转换为 numpy 数组
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        # 如果预测值中存在 NaN 或无穷大的值，需要进行处理
        y_pred = np.nan_to_num(y_pred, nan=0.5, posinf=1.0, neginf=0.0)

        # 存储真实标签和预测概率
        self.y_true.extend(y_true.tolist())
        self.y_pred.extend(y_pred.tolist())

    # 评估报告，返回一个字典，包含模型的多种评估指标
    def report(self):
        # 将预测值截断在 [0,1] 范围内，以防止超出界限
        y_pred = np.clip(self.y_pred, 0.0, 1.0)
        metrics_report = {}
        try:
            if len(self.y_true) == 0:
                metrics_report = {"acc": None, "auc": None, "mae": None, "rmse": None}
            else:
                # 计算准确率
                metrics_report["acc"] = metrics.accuracy_score(self.y_true, np.round(y_pred))
                # 计算 AUC 分数
                metrics_report["auc"] = metrics.roc_auc_score(self.y_true, y_pred)
                # 计算平均绝对误差
                metrics_report["mae"] = metrics.mean_absolute_error(self.y_true, y_pred)
                # 计算均方根误差
                metrics_report["rmse"] = metrics.mean_squared_error(self.y_true, y_pred) ** 0.5
        except Exception as e:
            metrics_report = {"acc": None, "auc": None, "mae": None, "rmse": None}

        return metrics_report
