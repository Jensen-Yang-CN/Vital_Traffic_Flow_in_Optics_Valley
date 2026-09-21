import pandas as pd
import numpy as np
df_raw = pd.read_csv('guanggu_traffic.csv')
df_raw['路段ID'] = df_raw['道路名称'] + "_" + df_raw['行驶方向']
speed_matrix = df_raw.pivot_table(
    index='采集时间',
    columns='路段ID',
    values='速度(km/h)',
    aggfunc='mean'
)

road_order = [
    '虎泉街_从民族大道到卓刀泉南路',
    '虎泉街_从雄楚大道到民族大道',
    '鲁磨路_从珞喻路到喻家山北路',
    '鲁磨路_从团山路到珞喻路',
    '珞喻路_从珞狮路到喻家湖路',
    '珞喻路_从民族大道到喻家湖路',
    '珞喻路_从喻家湖路到珞狮路',
    '珞喻路_从喻家湖路到民族大道',
    '珞喻路_从喻家湖路到卓刀泉立交',
    '珞喻路_从卓刀泉立交到民族大道',
    '珞喻路_喻家湖路附近',
    '民族大道_从珞喻路到南湖大道',
    '民族大道_从庙山立交到汤逊湖北路',
    '民族大道_从水蓝路到珞喻路',
    '民族大道_从汤逊湖北路到珞喻路',
    '民族大道_从雄楚大道到珞喻路'
]
speed_matrix = speed_matrix.reindex(columns=road_order)
speed_matrix = speed_matrix.ffill().bfill()
speed_matrix.to_csv('guanggu_speed.csv', index=False, header=False)
