import pandas as pd
import numpy as np

adj_matrix = np.load('guanggu_adj.npy')
# 转成 DataFrame 并导出为无表头的 CSV
pd.DataFrame(adj_matrix).to_csv('guanggu_adj.csv', index=False, header=False)
print("转换完成，可以用 Excel 打开 guanggu_adj.csv 啦！")