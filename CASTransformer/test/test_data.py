from ..data import KTData


def test_data():
    data_path = "data/assist09/train.txt"

    data = KTData(data_path, inputs=["pid", "q", "s", "d_correct", "d_skill_correct"])
    pid, q, s, d_correct, d_skill_correct = data[2].get("pid", "q", "s", "d_correct", "d_skill_correct")
    assert q.size() == s.size()     # q和s的尺寸必须相同
    assert d_correct.size() == s.size()     # d_correct和s的尺寸必须相同
    assert d_skill_correct.size() == s.size()   # d_skill_correct和s的尺寸必须相同

    # 初始化并验证第一个批次的数据点
    batch_size = 8
    data = KTData(data_path, inputs=["pid", "q", "s", "d_correct", "d_skill_correct"], batch_size=batch_size)
    q, s = next(iter(data)).get("q", "s")
    d_correct, d_skill_correct = next(iter(data)).get("d_correct", "d_skill_correct")

    assert q.size(0) == batch_size  # 检查 batch_size
    assert q.size() == s.size()  # q 和 s 的尺寸必须相同
    assert d_correct.size() == s.size()  # d_correct 和 s 的尺寸必须相同
    assert d_skill_correct.size() == s.size()  # d_skill_correct 和 s 的尺寸必须相同


    data_path = "data/assist17/train.txt"

    data = KTData(data_path, inputs=["q", "s", "pid", "it", "at", "d_correct", "d_skill_correct"])
    q, s = next(iter(data)).get("q", "s")
    assert q.size() == s.size()

    # 设置 batch_size
    batch_size = 4

    # 初始化 KTData 时包括新的字段
    data = KTData(
        data_path,
        inputs=["q", "s", "pid", "it", "at", "d_correct", "d_skill_correct"],
        batch_size=batch_size,
        shuffle=True,
    )

    q, s, at, d_correct, d_skill_correct = next(iter(data)).get("q", "s", "at", "d_correct", "d_skill_correct")

    assert q.size(0) == batch_size
    assert q.size() == s.size()
    assert q.size() == at.size()
    assert q.size() == d_correct.size()  # 验证 d_correct 的尺寸
    assert q.size() == d_skill_correct.size()  # 验证 d_skill_correct 的尺寸
