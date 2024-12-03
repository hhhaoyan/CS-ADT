import linecache    # 从Python源文件中高效地随机访问任意行。常用于调试和错误处理时，快速获取特定行的内容。
import math
import subprocess   # 允许用户生成新进程、连接其输入/输出/错误管道，并获得返回码。常用于运行外部命令和程序。
import sys          # 提供访问与Python解释器相关的变量和函数，常用于与解释器进行交互。

import torch
from torch.utils.data import DataLoader


# 定义Batch类， 用于批量处理数据
class Batch:
    def __init__(self, data, fields, seq_len=None):
        self.data = data
        # 存储字段名称
        self.fields = fields
        # 创建字段索引的字典，将字段名称映射到索引
        self.field_index = {f: i for i, f in enumerate(fields)}
        # 存储序列长度
        self.seq_len = seq_len

    # 获取指定字段的数据
    def get(self, *fields):
        # 获取数据长度
        L = len(self.data[0])
        valid_fields = [f for f in fields if f is not None]
        return (
            [self.data[self.field_index[f]] for f in valid_fields]
            if self.seq_len is None
            else [
                [
                    self.data[self.field_index[f]][
                        :, i * self.seq_len : (i + 1) * self.seq_len
                    ]
                    # math.ceil() 向上取整
                    # 按照指定的序列长度将数据分块
                    for i in range(math.ceil(L / self.seq_len))
                ]
                for f in fields
            ]
        )


class KTData:
    def __init__(
        self,
        data_path,  # 接受数据路径
        inputs,     # 输入字段
        batch_size=1,   # 批次大小
        seq_len=None,   # 序列长度
        shuffle=False,  # 是否打乱顺序
        num_workers=0,  # 工作线程数量
    ):
        # 读取数据文件，并按指定的字段数量分组
        self.data = Lines(data_path, group=len(inputs) + 1)
        # 创建DataLoader对象，用于批次数量加载
        self.loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=transform_batch,  # 数据整理函数
            num_workers=num_workers,
        )
        self.inputs = inputs
        self.seq_len = seq_len

    # 返回DataLoader的迭代器
    def __iter__(self):
        return iter(self.loader)

    # 返回数据的长度
    def __len__(self):
        return len(self.data)

    # 根据索引获取批次数据
    def __getitem__(self, index):
        return Batch(
            torch.tensor(
                [
                    [float(x) if '.' in x else int(x) for x in line.strip().split(",")]
                    for line in self.data[index][1:]
                ]
            ),
            self.inputs,
            self.seq_len,
        )

# transform 的批量操作
def transform_batch(batch):
    # 收集batch中的数据
    batch_data = [b.data for b in batch]
    # 合并配置，获取字段和序列长度f
    fields, seq_len = batch[0].fields, batch[0].seq_len

    # 转置数据, 以便分离各个序列
    batch = list(zip(*batch_data))
    # 填充序列, 使得所有序列具有相同的长度
    batch = [
        torch.nn.utils.rnn.pad_sequence(
            seqs,   # 要填充的序列
            batch_first=True,
            padding_value = -1
        )
        for seqs in batch
    ]

    # 返回一个新的batch对象,包含填充之后的数据, 字段和序列长度
    return Batch(batch, fields, seq_len)

class Lines:
    # 初始化方法，接收文件名、跳过行数、分组数和是否保留换行符等参数
    def __init__(self, filename, skip=0, group=1, preserve_newline=False):
        # 接受文件名
        self.filename = filename

        # 打开文件, 但不执行任何操作
        with open(filename):
            pass

        # 根据操作系统获取文件行数
        if sys.platform == "win32":
            # Windows平台使用Python计数行数
            linecount = sum(1 for _ in open(filename))
        else:
            # 非Windows平台使用shell命令统计行数，并将输出转换为整数
            output = subprocess.check_output(("wc -l " + filename).split())
            linecount = int(output.split()[0])
        self.length = (linecount - skip) // group
        self.skip = skip
        self.group = group
        self.preserve_newline = preserve_newline    # 是否保留换行符

    def __len__(self):
        return self.length

    # 返回一个迭代器, 用于逐行访问数据
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # 根据索引获取数据
    def __getitem__(self, item):
        # 跳过的行数加1，通常用于处理文件中的索引偏移
        d = self.skip + 1

        # 处理整数索引
        # 如果item是int类型, 则执行if内的代码.
        if isinstance(item, int):
            if item < len(self):
                if self.group == 1:     # 如果分组数为1，直接读取单行数据
                    line = linecache.getline(self.filename, item + d)
                    if not self.preserve_newline:
                        line = line.strip("\r\n")   # 去除行尾的换行符
                else:   # 如果分组数大于1，读取多行数据
                    line = [
                        linecache.getline(self.filename, d + item * self.group + k)
                        for k in range(self.group)
                    ]
                    if not self.preserve_newline:   # 不保留换行符
                        line = [l.strip("\r\n") for l in line]
                return line

        # 检查item是否是切片类型,如果是则进行一下操作
        # 处理切片索引
        elif isinstance(item, slice):

            # 将切片最小值赋值给low
            low = 0 if item.start is None else item.start
            # 将low限制在-len(self),len(self)-1 的范围内
            low = _clip(low, -len(self), len(self) - 1)

            # 将low变成正值
            if low < 0:
                low += len(self)

            # 设置high的值
            high = len(self) if item.stop is None else item.stop
            # 将high限制在-len(self),len(self)-1 的范围内
            high = _clip(high, -len(self), len(self))
            if high < 0:
                high += len(self)

            # 读取并处理文件内容
            # 这个部分和前面int类型的处理操作一样
            ls = []
            for i in range(low, high):
                if self.group == 1:
                    line = linecache.getline(self.filename, i + d)
                    if not self.preserve_newline:
                        line = line.strip("\r\n")
                else:
                    line = [
                        linecache.getline(self.filename, d + i * self.group + k)
                        for k in range(self.group)
                    ]
                    if not self.preserve_newline:
                        line = [l.strip("\r\n") for l in line]
                # 将line存储在ls列表里
                ls.append(line)

            return ls

        # 当出现错误的时候引发异常、
        raise IndexError

# 切片裁剪操作
# v: 需要裁剪的值。 low: 下限。 high: 上限
def _clip(v, low, high):
    if v < low:
        v = low
    if v > high:
        v = high
    return v
