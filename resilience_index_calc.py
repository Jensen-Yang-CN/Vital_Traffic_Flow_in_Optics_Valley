import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import models
import tasks
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
CHECKPOINT_PATH = r'lightning_logs/TGCN/version_10/checkpoints/epoch=247-step=21080.ckpt'
DATA_PATH = 'data/guanggu_speed.csv'
ADJ_PATH = 'data/guanggu_adj.csv'

adj_matrix = pd.read_csv(ADJ_PATH, header=None).values
model = models.TGCN(adj=adj_matrix, hidden_dim=32)
task = tasks.SupervisedForecastTask.load_from_checkpoint(CHECKPOINT_PATH, model=model)
task.eval()

speed_data = pd.read_csv(DATA_PATH, header=None).values.astype(np.float32)
max_val = speed_data.max()
def run_resilience_sim(start_idx, incident_node=8):
    base_input = speed_data[start_idx: start_idx + 18].copy() / max_val
    curr_input = torch.FloatTensor(base_input).unsqueeze(0)
    history = []

    with torch.no_grad():
        for t in range(12):
            pred = task(curr_input)
            next_val = pred[:, :, 0:1].transpose(1, 2)
            if t < 3:
                next_val[0, 0, incident_node] = 0.05

            history.append(next_val.squeeze().cpu().numpy())
            curr_input = torch.cat([curr_input[:, 1:, :], next_val], dim=1)

    return np.array(history) * max_val
morning_res = run_resilience_sim(start_idx=147)  # 早高峰时段
evening_res = run_resilience_sim(start_idx=1003)  # 晚高峰时段
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
node_to_show = 5

for i, (res, label) in enumerate(zip([morning_res, evening_res], ["早高峰", "晚高峰"])):
    baseline = res[0, node_to_show] + 1.5
    ax[i].fill_between(range(12), res[:, node_to_show], baseline, color='#ff9999', alpha=0.3, label='性能损失面积(PLA)')
    ax[i].plot(res[:, node_to_show], color='#d32f2f', marker='o', linewidth=2.5, markersize=6, label='受损恢复曲线')
    ax[i].axhline(y=baseline, color='#388e3c', linestyle='--', linewidth=2, label='理想性能基准')
    area_actual = np.trapz(res[:, node_to_show])
    area_ideal = baseline * 11
    R = area_actual / area_ideal
    ax[i].set_title(f"{label}路网韧性分析\n韧性指数 R = {R:.4f}", fontsize=15)
    ax[i].set_xlabel("事故发生后恢复时间 (每步 5 分钟)", fontsize=12)
    ax[i].set_ylabel("通行速度效能 (km/h)", fontsize=12)
    y_min = min(res[:, node_to_show]) - 5
    y_max = baseline + 5
    ax[i].set_ylim(y_min, y_max)

    ax[i].legend(loc='lower right', fontsize=11)
    ax[i].grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig('Guanggu_Resilience_Triangle_Official.png', dpi=300)
print("✅ 早晚高峰韧性三角图已生成，并保存为 Guanggu_Resilience_Triangle_Official.png")
plt.show()