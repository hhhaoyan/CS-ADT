from dataclasses import dataclass


@dataclass
# 装饰器，用于简化类的定义，自动为类生成一些常用的特殊方法。
class Configuration:
    # 创建数据类，但是没有任何属性和方法
    pass


# 用于加载TOML文件
# str：表示路径，并返回一个configuration类型的对象
def load_toml(path: str) -> Configuration:
    pass
