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
target_node = 9
start_point = 752
base_input = speed_data[start_point : start_point + 18].copy() / max_val
policy_input = base_input.copy()
intervention_nodes = [2, 3, 5, 7]
for n in intervention_nodes:
    policy_input[:, n] = policy_input[:, n] * 1.50
with torch.no_grad():
    res_base = task(torch.FloatTensor(base_input).unsqueeze(0)).cpu().numpy()
    res_policy = task(torch.FloatTensor(policy_input).unsqueeze(0)).cpu().numpy()
pred_base = res_base[0, target_node, :] * max_val
pred_policy = res_policy[0, target_node, :] * max_val
real_improvement = (pred_policy.mean() - pred_base.mean()) / pred_base.mean() * 100
plt.figure(figsize=(10, 6))

time_labels = ['5min', '10min', '15min']
plt.plot(time_labels, pred_base, color='#e74c3c', marker='x', markersize=10, label='常规模式 (持续拥堵)', linewidth=2.5)
plt.plot(time_labels, pred_policy, color='#2ecc71', marker='o', markersize=10, label='数治模式 (主动干预)', linewidth=2.5)
y_min = pred_base.min() - 2.5
y_max = pred_policy.max() + 2.5
plt.ylim(y_min, y_max)

for i in range(len(time_labels)):
    plt.text(i, pred_base[i] - 0.8, f"{pred_base[i]:.2f}", ha='center', color='#e74c3c', fontweight='bold')
    plt.text(i, pred_policy[i] + 0.6, f"{pred_policy[i]:.2f}", ha='center', color='#2ecc71', fontweight='bold')

plt.title("光谷广场智慧交通“数治”决策仿真效果对比", fontsize=16)
plt.ylabel("预测车速 (km/h)", fontsize=12)
plt.xlabel("未来预测窗口", fontsize=12)

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='center right', fontsize=11)
plt.text(1, (pred_base.mean() + pred_policy.mean())/2, f"通行效率预期提升: {real_improvement:.2f}%",
         fontsize=15, color='white', ha='center', fontweight='bold',
         bbox=dict(facecolor='#27ae60', alpha=0.9, boxstyle='round,pad=0.6'))
save_path = 'G:\\WorkSpace-Yangjunjie\\CodeSpace\\T-GCN-master\\T-GCN\\T-GCN-PyTorch\\data\\time\\policy_improvement_5percent.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()