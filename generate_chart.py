import matplotlib.pyplot as plt
import os

epochs = [20, 40, 60, 80, 100]
nme = [0.056769, 0.048924, 0.047724, 0.047536, 0.047495]
auc = [0.436994, 0.512630, 0.524249, 0.526152, 0.526152]

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('NME', color=color)
ax1.plot(epochs, nme, marker='o', color=color, linewidth=2, label='NME (Lower is better)')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('AUC', color=color)
ax2.plot(epochs, auc, marker='s', color=color, linewidth=2, label='AUC (Higher is better)')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Optimization Progress on COFW Dataset')
plt.grid(True, alpha=0.3)
plt.savefig('/Users/tanya/Desktop/中期/optimization_progress.png')
print("Chart saved")
