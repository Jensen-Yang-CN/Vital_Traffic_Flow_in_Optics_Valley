import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import models
import tasks
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
INCIDENT_NODE = 8
CONTROL_NODE = 7
def simulate_intervention(start_idx, alpha_control):
    base_input = speed_data[start_idx: start_idx + 18].copy() / max_val
    curr_input = torch.FloatTensor(base_input).unsqueeze(0)
    history = []
    with torch.no_grad():
        for t in range(12):
            pred = task(curr_input)
            next_val = pred[:, :, 0:1].transpose(1, 2)
            if t < 6:
                next_val[0, 0, INCIDENT_NODE] = 0.01
            if t < 6 and alpha_control > 0:
                next_val[0, 0, CONTROL_NODE] = alpha_control
            history.append(next_val.squeeze().cpu().numpy())
            curr_input = torch.cat([curr_input[:, 1:, :], next_val], dim=1)

    return np.array(history) * max_val
alphas = np.arange(0.0, 1.1, 0.1)  # 从 0.0 到 1.0，步长 0.1
network_avg_speeds = []
efficiency_gains = []
baseline_history = simulate_intervention(start_idx=21, alpha_control=0.0)
baseline_avg_speed = np.mean(np.delete(baseline_history, INCIDENT_NODE, axis=1))

for alpha in alphas:
    if alpha == 0.0:
        network_avg_speeds.append(baseline_avg_speed)
        efficiency_gains.append(0.0)
        continue
    interv_history = simulate_intervention(start_idx=21, alpha_control=alpha)
    avg_speed = np.mean(np.delete(interv_history, INCIDENT_NODE, axis=1))
    network_avg_speeds.append(avg_speed)
    gain_pct = (avg_speed - baseline_avg_speed) / baseline_avg_speed * 100
    efficiency_gains.append(gain_pct)
plt.figure(figsize=(10, 6))
plt.plot(alphas, efficiency_gains, marker='o', linestyle='-', color='#2ca02c', linewidth=2.5, markersize=8)
plt.axvline(x=0.6, color='red', linestyle='--', label='推演寻优拐点 (最优干预阈值 α=0.6)')
plt.title('光谷路网前馈式管控干预强度与边际效益寻优分析', fontsize=15)
plt.xlabel('主动管控强度系数 α (对应绿信比补偿幅度)', fontsize=13)
plt.ylabel('路网整体通行效率提升比例 (%)', fontsize=13)
plt.xticks(alphas)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11)

for i, txt in enumerate(efficiency_gains):
    if i % 2 == 0 or i == len(efficiency_gains) - 1:  # 隔一个标一次，防止太挤
        plt.annotate(f"{txt:.1f}%", (alphas[i], efficiency_gains[i]), textcoords="offset points", xytext=(0, 10),
                     ha='center')

plt.tight_layout()
plt.savefig('Control_Optimization_Curve.png', dpi=300)
print("边际效益寻优曲线已保存为 Control_Optimization_Curve.png")
plt.show()