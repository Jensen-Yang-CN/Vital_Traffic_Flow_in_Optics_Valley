import pandas as pd
import os
import sys


def clean_traffic_data(input_file):
    """
    清洗交通数据CSV文件
    参数:
        input_file: 输入的CSV文件路径
    输出:
        生成 {input_file_base}_cleaned.csv 文件
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：文件 {input_file} 不存在！")
        return

    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 第一轮筛选：只保留指定的道路名称
    target_roads = ['民族大道', '珞喻路', '虎泉街', '鲁磨路']
    df_filtered = df[df['道路名称'].isin(target_roads)].copy()

    # 第二轮筛选规则：每个道路名称允许保留的行驶方向
    direction_rules = {
        '虎泉街': ['从民族大道到卓刀泉南路', '从雄楚大道到民族大道', '从卓刀泉南路到民族大道'],
        '鲁磨路': ['从珞喻路到喻家山北路', '从团山路到珞喻路'],
        '珞喻路': ['从珞狮路到喻家湖路', '从喻家湖路到珞狮路'],
        '民族大道': ['从汤逊湖北路到珞喻路', '从雄楚大道到珞喻路', '从珞喻路到南湖大道', '从珞喻路到大学园路','从珞喻路到雄楚大道']
    }

    # 构建筛选掩码
    mask = pd.Series([False] * len(df_filtered), index=df_filtered.index)
    for road, directions in direction_rules.items():
        road_mask = (df_filtered['道路名称'] == road) & (df_filtered['行驶方向'].isin(directions))
        mask |= road_mask

    df_cleaned = df_filtered[mask]

    # 生成输出文件名（源文件名_cleaned.csv）
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_cleaned{ext}"

    # 保存结果（如果文件已存在，提示并覆盖？但用户要求不覆盖之前清洗过的文件，这里覆盖的是同一源文件生成的同名文件，可以覆盖）
    # 为了安全，可以询问，但默认覆盖（因为每次清洗同一源文件，期望更新输出）
    df_cleaned.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"清洗完成！")
    print(f"原始数据    : {len(df)} 行")
    print(f"第一轮筛选后: {len(df_filtered)} 行")
    print(f"最终保留    : {len(df_cleaned)} 行")
    print(f"结果已保存至: {output_file}")


if __name__ == "__main__":
    # 从命令行参数获取输入文件，如果没有则交互输入
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = input("请输入要清洗的CSV文件路径: ").strip()

    clean_traffic_data(input_file)