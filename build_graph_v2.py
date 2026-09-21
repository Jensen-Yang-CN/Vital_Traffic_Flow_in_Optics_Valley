import pandas as pd
import numpy as np
import re

# 1. ⚠️ 修改重点：根据最新 27 号数据，扩充为 13 个节点
road_nodes = [
    # 虎泉街 (3个)
    {'道路名称': '虎泉街', '行驶方向': '从民族大道到卓刀泉南路'},
    {'道路名称': '虎泉街', '行驶方向': '从雄楚大道到民族大道'},
    {'道路名称': '虎泉街', '行驶方向': '从卓刀泉南路到民族大道'}, # 27号数据中发现的

    # 鲁磨路 (2个)
    {'道路名称': '鲁磨路', '行驶方向': '从珞喻路到喻家山北路'},
    {'道路名称': '鲁磨路', '行驶方向': '从团山路到珞喻路'},

    # 珞喻路 (2个)
    {'道路名称': '珞喻路', '行驶方向': '从珞狮路到喻家湖路'},
    {'道路名称': '珞喻路', '行驶方向': '从喻家湖路到珞狮路'},

    # 民族大道 (6个) - 补充了27号数据中变化的描述
    {'道路名称': '民族大道', '行驶方向': '从汤逊湖北路到珞喻路'},
    {'道路名称': '民族大道', '行驶方向': '从雄楚大道到珞喻路'},
    {'道路名称': '民族大道', '行驶方向': '从珞喻路到南湖大道'},
    {'道路名称': '民族大道', '行驶方向': '从珞喻路到大学园路'},   # 27号数据中发现的
    {'道路名称': '民族大道', '行驶方向': '从庙山立交到汤逊湖北路'},
    {'道路名称': '民族大道', '行驶方向': '从水蓝路到珞喻路'}
]

df = pd.DataFrame(road_nodes)
df['路段ID'] = df['道路名称'] + "_" + df['行驶方向']

def parse_direction(text):
    match = re.search(r'从(.*)到(.*)', text)
    if match:
        return match.group(1), match.group(2)
    return None, None

df[['起点', '终点']] = df.apply(lambda row: pd.Series(parse_direction(row['行驶方向'])), axis=1)

# 2. 构建邻接矩阵 (13 x 13)
nodes = df['路段ID'].tolist()
num_nodes = len(nodes)
adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

print(f"🚀 正在构建光谷核心区拓扑网络 (节点数: {num_nodes})...")

for i, row_i in df.iterrows():
    for j, row_j in df.iterrows():
        if i == j: continue

        # 核心逻辑 A: 流入转盘 (民族大道/鲁磨路/虎泉街进站) -> 流向珞喻路/鲁磨路
        if row_i['终点'] == '珞喻路' or row_i['终点'] == '民族大道':
            if row_j['道路名称'] in ['珞喻路', '鲁磨路', '民族大道'] and row_j['起点'] in ['珞狮路', '喻家湖路', '珞喻路']:
                adj_matrix[i, j] = 1

        # 核心逻辑 B: 珞喻路横穿转盘
        if row_i['道路名称'] == '珞喻路' and row_j['道路名称'] == '珞喻路':
            if row_i['终点'] == row_j['起点'] or row_i['起点'] == row_j['终点']:
                adj_matrix[i, j] = 1

# 导出
np.save('guanggu_adj.npy', adj_matrix)
# 这一行就是生成你要的“户口本”
df['路段ID'].to_csv('guanggu_node_order.txt', index=False, header=False)

print("\n✅ 邻接矩阵已更新为 13x13 维度，并保存为 guanggu_adj.npy")
print("✅ 法定节点顺序已更新至 guanggu_node_order.txt")