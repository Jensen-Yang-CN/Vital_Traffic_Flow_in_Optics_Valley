import pandas as pd
files = [
    'guanggu_traffic_4_24_cleaned.csv',
    'guanggu_traffic_4_25_cleaned.csv',
    'guanggu_traffic_4_26_cleaned.csv',
    'guanggu_traffic_4_27_cleaned.csv',
    'guanggu_traffic_4_28_cleaned.csv',
    'guanggu_traffic_4_29_cleaned.csv',
    'guanggu_traffic_4_30_cleaned.csv'
]
def smart_read_csv(file_path):
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2132', 'ansi']
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"成功读取 {file_path} (编码: {enc})，行数: {len(df)}")
            return df
        except UnicodeDecodeError:
            continue
    return None
all_data = []
for f in files:
    df = smart_read_csv(f)
    if df is not None:
        all_data.append(df)
if all_data:
    df_combined = pd.concat(all_data, ignore_index=True)
    save_name = 'guanggu_traffic_ALL_combined_V3_latest.csv'
    df_combined.to_csv(save_name, index=False, encoding='utf-8-sig')
