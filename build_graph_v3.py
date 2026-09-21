import pandas as pd
import numpy as np
import re
road_nodes = [
    # 虎泉街 (3个)
    {'道路名称': '虎泉街', '行驶方向': '从民族大道到卓刀泉南路'},
    {'道路名称': '虎泉街', '行驶方向': '从雄楚大道到民族大道'},
    {'道路名称': '虎泉街', '行驶方向': '从卓刀泉南路到民族大道'},

    # 鲁磨路 (2个)
    {'道路名称': '鲁磨路', '行驶方向': '从团山路到珞喻路'},
    {'道路名称': '鲁磨路', '行驶方向': '从珞喻路到喻家山北路'},

    # 珞喻路 (2个)
    {'道路名称': '珞喻路', '行驶方向': '从珞狮路到喻家湖路'},
    {'道路名称': '珞喻路', '行驶方向': '从喻家湖路到珞狮路'},

    # 民族大道 (5个)
    {'道路名称': '民族大道', '行驶方向': '从汤逊湖北路到珞喻路'},
    {'道路名称': '民族大道', '行驶方向': '从雄楚大道到珞喻路'},
    {'道路名称': '民族大道', '行驶方向': '从珞喻路到南湖大道'},
    {'道路名称': '民族大道', '行驶方向': '从珞喻路到大学园路'},
    {'道路名称': '民族大道', '行驶方向': '从珞喻路到雄楚大道'}
]
df = pd.DataFrame(road_nodes)
df['路段ID'] = df['道路名称'] + "_" + df['行驶方向']
def parse_direction(text):
    match = re.search(r'从(.*)到(.*)', text)
    if match:
        return match.group(1), match.group(2)
    return None, None
df[['起点', '终点']] = df.apply(lambda row: pd.Series(parse_direction(row['行驶方向'])), axis=1)
nodes = df['路段ID'].tolist()
num_nodes = len(nodes)
adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
print(f"正在构建光谷核心区拓扑网络 (最终节点数: {num_nodes})...")
for i, row_i in df.iterrows():
    for j, row_j in df.iterrows():
        if i == j: continue
        if row_i['终点'] == row_j['起点']:
            adj_matrix[i, j] = 1
        if row_i['终点'] == '珞喻路' and row_j['起点'] == '珞喻路':
            adj_matrix[i, j] = 1
np.save('guanggu_adj.npy', adj_matrix)
df['路段ID'].to_csv('guanggu_node_order.txt', index=False, header=False)
