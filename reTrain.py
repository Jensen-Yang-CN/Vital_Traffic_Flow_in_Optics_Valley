import pandas as pd
import numpy as np
import os

def smart_read(file_path):
    for enc in ['gbk', 'utf-8-sig', 'utf-8', 'ansi']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except:
            continue
    raise Exception("实在读不动这个文件，请检查编码")
df = smart_read('guanggu_traffic_ALL_combined_V3_latest.csv')
df['路段ID'] = df['道路名称'] + "_" + df['行驶方向']
with open('guanggu_node_order.txt', 'r', encoding='utf-8') as f:
    node_order = [line.strip() for line in f.readlines()]
speed_matrix = df.pivot_table(
    index='采集时间',
    columns='路段ID',
    values='速度(km/h)',
    aggfunc='mean'
)
speed_matrix = speed_matrix.reindex(columns=node_order)
speed_matrix = speed_matrix.ffill().bfill()
if speed_matrix.isnull().values.any():
    speed_matrix = speed_matrix.fillna(0) # 或者用 speed_matrix.mean()
if (speed_matrix.max() == 0).any():
    speed_matrix = speed_matrix.replace(0, 1.0) # 给个基础速度防止除以0
if not os.path.exists('data'): os.makedirs('data')
speed_matrix.to_csv('data/guanggu_speed.csv', index=False, header=False)

if os.path.exists('guanggu_adj.npy'):
    adj = np.load('guanggu_adj.npy')
    pd.DataFrame(adj).to_csv('data/guanggu_adj.csv', index=False, header=False)
