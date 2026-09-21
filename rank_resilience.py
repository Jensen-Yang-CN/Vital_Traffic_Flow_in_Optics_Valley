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
node_names = {
    0: "虎泉街_从民族大道到卓刀泉南路",
    1: "虎泉街_从雄楚大道到民族大道",
    2: "虎泉街_从卓刀泉南路到民族大道",
    3: "鲁磨路_从团山路到珞喻路",
    4: "鲁磨路_从珞喻路到喻家山北路",
    5: "珞喻路_从珞狮路到喻家湖路",
    6: "珞喻路_从喻家湖路到珞狮路",
    7: "民族大道_从汤逊湖北路到珞喻路",
    8: "民族大道_从雄楚大道到珞喻路",  # 事故冲击源
    9: "民族大道_从珞喻路到南湖大道",
    10: "民族大道_从珞喻路到大学园路",
    11: "民族大道_从珞喻路到雄楚大道"
}


def run_simulation(start_idx, incident_node=None):
    base_input = speed_data[start_idx: start_idx + 18].copy() / max_val
    curr_input = torch.FloatTensor(base_input).unsqueeze(0)
    history = []

    with torch.no_grad():
        for t in range(12):
            pred = task(curr_input)
            next_val = pred[:, :, 0:1].transpose(1, 2)
            if incident_node is not None and t < 3:
                next_val[0, 0, incident_node] = 0.05

            history.append(next_val.squeeze().cpu().numpy())
            curr_input = torch.cat([curr_input[:, 1:, :], next_val], dim=1)

    return np.array(history) * max_val
SIM_START_IDX = 752
baseline_res = run_simulation(start_idx=SIM_START_IDX, incident_node=None)
cascade_res = run_simulation(start_idx=SIM_START_IDX, incident_node=8)
resilience_scores = {}
for i in range(12):
    if i == 8:
        continue
    v_ideal = baseline_res[:, i]
    v_actual = cascade_res[:, i]
    R_index = np.sum(v_actual) / np.sum(v_ideal)
    name = node_names.get(i, f"Node_{i}")
    resilience_scores[name] = R_index
sorted_resilience = sorted(resilience_scores.items(), key=lambda x: x[1], reverse=True)
names = [x[0] for x in sorted_resilience]
scores = [x[1] for x in sorted_resilience]
plt.figure(figsize=(14, 9))
min_score, max_score = min(scores), max(scores)
normalized_scores = (np.array(scores) - min_score) / (max_score - min_score + 1e-5)
colors = plt.cm.RdYlGn(normalized_scores)
bars = plt.barh(names, scores, color=colors, edgecolor='black', linewidth=0.5)
for bar in bars:
    width = bar.get_width()
    if width < min_score + (max_score - min_score) * 0.15:
        plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                 f'{width:.4f}', ha='left', va='center', color='black', fontweight='bold', fontsize=11)
    else:
        plt.text(width - 0.005, bar.get_y() + bar.get_height() / 2,
                 f'{width:.4f}', ha='right', va='center', color='black', fontweight='bold', fontsize=11)
plt.axvline(x=1.0, color='gray', linestyle='--', linewidth=2, label='理论完美韧性 (R=1.0)')
plt.xlabel('路网韧性指数 R (面积占比，越接近 1.0 抗冲击能力越强)', fontsize=14)
plt.ylabel('光谷广场微循环节点', fontsize=14)
plt.title('光谷路网各节点抗冲击韧性排名 (民族大道_从雄楚大道到珞喻路 事故级联推演)', fontsize=16)

plt.xlim(min(scores) - 0.02, max(scores) + 0.02)
plt.gca().invert_yaxis()  # 让韧性最高的排在最上面
plt.legend(loc='lower left')
plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('Guanggu_Resilience_Ranking_Official.png', dpi=300)
print("✅ 韧性排名图已生成并保存为 Guanggu_Resilience_Ranking_Official.png")
plt.show()