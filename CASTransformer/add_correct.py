import os
import csv


# 读取第一组数据，包含用户ID，概念(skill_ids)，回答情况(correct_responses)，题目(item_ids)，开始时间(start_times)，结束时间(end_times)
def read_first_dataset(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
        # 第一行为用户ID，第二行为概念ID，第三行为回答情况，第四行为题目ID，第五行为开始时间，第六行为结束时间
        user_id = lines[0]  # 用户ID
        skill_ids = list(map(int, lines[1].split(',')))  # 概念ID
        correct_responses = list(map(int, lines[2].split(',')))  # 回答情况
        item_ids = list(map(int, lines[3].split(',')))  # 题目ID
        start_times = list(map(int, lines[4].split(',')))  # 开始时间
        end_times = list(map(int, lines[5].split(',')))  # 结束时间

        return user_id, skill_ids, correct_responses, item_ids, start_times, end_times


# 读取第二组数据，包含 item_id, skill_id, d_correct, d_skill_correct
def read_second_dataset(file_path):
    second_dataset = {
        'item': {},  # 用于存储 item_id -> d_correct
        'skill': {}  # 用于存储 skill_id -> d_skill_correct
    }

    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在，跳过。")
        return second_dataset

    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')  # 假设使用制表符分隔
        headers = next(reader)  # 跳过表头

        for row in reader:
            try:
                item_id = int(row[1])  # item_id 位于第二列
                skill_id = int(row[4])  # skill_id 位于第五列
                d_correct = float(row[6])  # d_correct 位于第九列
                d_skill_correct = float(row[7])  # d_skill_correct 位于第十列

                # 存储 item_id 和 skill_id 对应的 d_correct 和 d_skill_correct
                second_dataset['item'][item_id] = d_correct
                second_dataset['skill'][skill_id] = d_skill_correct
            except (ValueError, IndexError):
                print(f"跳过无效数据行: {row}")

    return second_dataset


# 合并数据，将 d_correct 和 d_skill_correct 分别添加到新的输出文件中
def merge_datasets(first_dataset, second_dataset, backup_dataset):
    user_id, skill_ids, correct_responses, item_ids, start_times, end_times = first_dataset
    d_corrects = []  # 用于存储 d_correct 数据
    d_skill_corrects = []  # 用于存储 d_skill_correct 数据

    # 根据 item_id 查找 d_correct
    for item_id in item_ids:
        if item_id in second_dataset['item']:
            d_correct = second_dataset['item'][item_id]
        elif item_id in backup_dataset['item']:
            d_correct = backup_dataset['item'][item_id]  # 如果在 test.txt 找不到，则从 train.txt 中查找
        else:
            d_correct = 0  # 如果找不到对应的 d_correct，设为 0
        d_corrects.append(d_correct)

    # 根据 skill_id 查找 d_skill_correct
    for skill_id in skill_ids:
        if skill_id in second_dataset['skill']:
            d_skill_correct = second_dataset['skill'][skill_id]
        elif skill_id in backup_dataset['skill']:
            d_skill_correct = backup_dataset['skill'][skill_id]  # 如果在 test.txt 找不到，则从 train.txt 中查找
        else:
            d_skill_correct = 0  # 如果找不到对应的 d_skill_correct，设为 0
        d_skill_corrects.append(d_skill_correct)

    return user_id, skill_ids, correct_responses, item_ids, start_times, end_times, d_corrects, d_skill_corrects


# 读取并处理第一组数据的所有用户
def read_and_process_first_dataset(file_path):
    datasets = []
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
        # 处理每个用户的数据 (每6行一个用户的数据)
        i = 0
        while i < len(lines):
            if i + 5 >= len(lines):
                print(f"文件 {file_path} 中的数据格式不完整，从第 {i} 行开始的数据将被忽略。")
                break
            user_id = lines[i]
            skill_ids = list(map(int, lines[i + 1].split(',')))  # 第二行为概念ID
            correct_responses = list(map(int, lines[i + 2].split(',')))  # 第三行为回答情况
            item_ids = list(map(int, lines[i + 3].split(',')))  # 第四行为题目ID
            start_times = list(map(int, lines[i + 4].split(',')))  # 第五行为开始时间
            end_times = list(map(int, lines[i + 5].split(',')))  # 第六行为结束时间
            datasets.append((user_id, skill_ids, correct_responses, item_ids, start_times, end_times))
            i += 6  # 每个用户占6行
    return datasets


# 保存合并后的数据，支持多个用户
def save_merged_data(output_path, merged_datasets):
    with open(output_path, 'w') as f:
        for merged_data in merged_datasets:
            user_id, skill_ids, correct_responses, item_ids, start_times, end_times, d_corrects, d_skill_corrects = merged_data
            f.write(f"{user_id}\n")  # 写入用户ID
            f.write(','.join(map(str, skill_ids)) + '\n')  # 写入概念ID
            f.write(','.join(map(str, correct_responses)) + '\n')  # 写入作答情况
            f.write(','.join(map(str, item_ids)) + '\n')  # 写入题目ID
            f.write(','.join(map(str, start_times)) + '\n')  # 写入开始时间
            f.write(','.join(map(str, end_times)) + '\n')  # 写入结束时间
            f.write(','.join(map(str, d_corrects)) + '\n')  # 写入 d_correct
            f.write(','.join(map(str, d_skill_corrects)) + '\n')  # 写入 d_skill_correct


# 处理单个文件，优先在 test.txt 中查找，如果找不到则在 train.txt 中查找
def process_file(first_file_path, second_test_file_path, second_train_file_path, output_file_path):
    # 读取多个用户的第一组数据
    first_datasets = read_and_process_first_dataset(first_file_path)

    # 读取 test.txt 数据
    second_dataset = read_second_dataset(second_test_file_path)

    # 读取 train.txt 数据作为备用
    backup_dataset = read_second_dataset(second_train_file_path)

    merged_datasets = []
    # 合并每个用户的数据
    for first_dataset in first_datasets:
        merged_data = merge_datasets(first_dataset, second_dataset, backup_dataset)
        merged_datasets.append(merged_data)

    # 保存合并后的数据
    save_merged_data(output_file_path, merged_datasets)

    print(f"处理完成: {first_file_path}，结果保存为: {output_file_path}")


# 批量处理 test.txt 和 train.txt 文件，互为备用数据源
def batch_process_datasets(first_dataset_dir, second_dataset_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 处理 test.txt 文件，优先在 test.txt 中查找，找不到则在 train.txt 中查找
    process_file(
        os.path.join(first_dataset_dir, 'test.txt'),
        os.path.join(second_dataset_dir, 'test.txt'),
        os.path.join(second_dataset_dir, 'train.txt'),
        os.path.join(output_dir, 'test.txt')
    )

    # 处理 train.txt 文件，优先在 train.txt 中查找，找不到则在 test.txt 中查找
    process_file(
        os.path.join(first_dataset_dir, 'train.txt'),
        os.path.join(second_dataset_dir, 'train.txt'),
        os.path.join(second_dataset_dir, 'test.txt'),
        os.path.join(output_dir, 'train.txt')
    )


# 主函数
def main():
    first_dataset_dir = r'D:\DMD-Transformer\data_primitive\assist17'  # 第一组数据的目录路径
    second_dataset_dir = r'D:\DMD-Transformer\data_difficulty\assist17'  # 第二组数据的目录路径
    output_dir = r'D:\DMD-Transformer\output\assist17'  # 输出目录路径

    # 批量处理 test.txt 和 train.txt 数据
    batch_process_datasets(first_dataset_dir, second_dataset_dir, output_dir)


if __name__ == "__main__":
    main()
