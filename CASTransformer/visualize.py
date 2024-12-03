import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# 给定轴对象上绘制热力图。
def heat_map(ax, alpha, cmap="hot", xticks=None, yticks=None):

    im = ax.pcolormesh(alpha, edgecolors="grey", linewidth=0.4, cmap=cmap)

    # 反转 y 轴，使得 y 轴从上到下增加。
    ax.invert_yaxis()

    # 设置 x 轴的刻度
    if xticks is None:
        # 轴的范围：[0, alpha.shape[1]+1] 步长为 5
        ax.set_xticks(np.arange(0, alpha.shape[1] + 1, 5))
    else:
        ax.set_xticks(xticks)

    # 设置 y 轴的刻度
    if yticks is None:
        # 轴的范围：[0, alpha.shape[1]+1] 步长为 5
        ax.set_yticks(np.arange(0, alpha.shape[0] + 1, 5))
    else:
        ax.set_yticks(yticks)

    # 最后返回绘制的图像对象 im
    return im


def trace_map(y, q, s, span=range(5), k_color=None, text_label=False, figsize=(22, 3)):
    """
    y: [1, selected_features] tensor
    q: [34] tensor
    s: [34] tensor
    span: range object, corresponds to the selected features
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 确保 y 是二维张量
    if y.dim() != 2:
        raise ValueError(f"Expected y to be a 2D tensor, but got {y.dim()}D tensor")

    # 将 y 转换为 NumPy 数组
    y_np = y.detach().numpy()  # 形状 [1, 5]

    # 绘制伪彩色网格
    im = ax.pcolormesh(
        y.detach().numpy(), edgecolors="w", linewidth=0.4, cmap="RdYlGn", clim=(0, 1)
    )

    ax.invert_yaxis()
    ax.set_yticks(np.arange(0, y_np.shape[0] + 1, 5))

    # 添加颜色条
    plt.colorbar(im, ax=ax, location="right")

    # 如果未提供 k_color，自动生成
    if k_color is None:
        knows = list(set(q[0, span].tolist()))
        cmap = matplotlib.cm.get_cmap("tab20")
        k_color = {k: cmap(i) for i, k in enumerate(knows)}

    # 圆形标记的偏移量
    x_offset = 0.5
    y_offset = -0.6

    for x, i in enumerate(span):
        if i == 0:
            continue
        q_ = q[0, i - 1].item()
        s_ = s[0, i - 1].item()
        ax.add_patch(
            plt.Circle((x + x_offset, y_offset), 0.4, color=k_color.get(q_, "gray"), clip_on=False)
        )
        if s_ == 1:
            ax.add_patch(
                plt.Circle(
                    (x + x_offset, y_offset), 0.2, color="w", zorder=100, clip_on=False
                )
            )

    # 添加文本标签
    if text_label:
        label = []
        for i in span:
            if i == 0:
                label.append("-")
            else:
                label.append(f"{q[0, i - 1].item()}-{s[0, i - 1].item()}")
        ax.set_xticks(np.arange(0.5, len(span)), label=label)

    return fig