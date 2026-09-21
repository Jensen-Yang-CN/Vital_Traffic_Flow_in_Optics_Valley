import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import models
import tasks
import os
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
HIDDEN_DIM = 32
SEQ_LEN = 18
PRE_LEN = 3
N_NODES = 12
CHECKPOINT_PATH = r'lightning_logs/TGCN/version_10/checkpoints/epoch=247-step=21080.ckpt'
DATA_PATH = 'data/guanggu_speed.csv'
ADJ_PATH = 'data/guanggu_adj.csv'

adj_matrix = pd.read_csv(ADJ_PATH, header=None).values
model = models.TGCN(adj=adj_matrix, hidden_dim=HIDDEN_DIM)
task = tasks.SupervisedForecastTask.load_from_checkpoint(CHECKPOINT_PATH, model=model)
task.eval()
speed_data = pd.read_csv(DATA_PATH, header=None).values.astype(np.float32)
max_val = speed_data.max()
start_point = 752
base_input = speed_data[start_point: start_point + SEQ_LEN].copy() / max_val
impacted_input = base_input.copy()
impacted_node = 8
impacted_input[:, impacted_node] = 0.05


steps_to_predict = 6
cascade_results = []
baseline_results = []

curr_input_impact = torch.FloatTensor(impacted_input).unsqueeze(0)
curr_input_base = torch.FloatTensor(base_input).unsqueeze(0)

with torch.no_grad():
    for _ in range(steps_to_predict):
        pred_base = task(curr_input_base)
        next_val_base = pred_base[:, :, 0:1].transpose(1, 2)
        baseline_results.append(next_val_base.squeeze().cpu().numpy())
        curr_input_base = torch.cat([curr_input_base[:, 1:, :], next_val_base], dim=1)
        pred_impact = task(curr_input_impact)
        next_val_impact = pred_impact[:, :, 0:1].transpose(1, 2)
        next_val_impact_fixed = next_val_impact.clone()
        next_val_impact_fixed[0, 0, impacted_node] = 0.05
        cascade_results.append(next_val_impact.squeeze().cpu().numpy())
        curr_input_impact = torch.cat([curr_input_impact[:, 1:, :], next_val_impact_fixed], dim=1)
baseline_res = np.array(baseline_results) * max_val
cascade_res = np.array(cascade_results) * max_val
fig, ax = plt.subplots(1, 2, figsize=(18, 7))
node_names = {
    5: "珞喻路_从珞狮路到喻家湖路 (直接关联)",
    0: "虎泉街_从民族大道到卓刀泉南路 (间接关联/拓扑隔离)"
}

for idx, node in enumerate([5, 0]):
    mean_baseline = np.mean(baseline_res[:, node])
    mean_cascade = np.mean(cascade_res[:, node])
    avg_loss = (mean_baseline - mean_cascade) / mean_baseline * 100

    ax[idx].plot(baseline_res[:, node], 'g-o', label='正常状态 (基准)', linewidth=2.5, markersize=8)
    ax[idx].plot(cascade_res[:, node], 'r--x', label=f'受冲击状态 (Node {impacted_node} 事故)', linewidth=2.5, markersize=8)

    ax[idx].set_title(f"{node_names[node]}\n平均通行效率损失: {avg_loss:.2f}%", fontsize=15)
    ax[idx].set_xlabel("预测时长 (每步5分钟)", fontsize=13)
    ax[idx].set_ylabel("预测速度 (km/h)", fontsize=13)
    ax[idx].legend(fontsize=11)
    ax[idx].grid(True, linestyle='--', alpha=0.5)
    y_min = min(np.min(baseline_res[:, node]), np.min(cascade_res[:, node]))
    y_max = max(np.max(baseline_res[:, node]), np.max(cascade_res[:, node]))
    ax[idx].set_ylim(y_min - 2, y_max + 2)

plt.tight_layout()
plt.savefig('guanggu_cascade_comparison_Official.png', dpi=300)
print("✅ 级联失效推演完成！对比图已保存为 guanggu_cascade_comparison_Official.png")
plt.show()