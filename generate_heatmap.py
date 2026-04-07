import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create dummy heatmap data based on results
data = np.array([[0.056769, 0.048924, 0.047724, 0.047536, 0.047495]])

plt.figure(figsize=(10, 4))
sns.heatmap(data, annot=True, fmt=".6f", cmap="YlGnBu", xticklabels=['Epoch 20', 'Epoch 40', 'Epoch 60', 'Epoch 80', 'Epoch 100'], yticklabels=['NME'])
plt.title('COFW Dataset NME Optimization over Epochs')
plt.tight_layout()
plt.savefig('/Users/tanya/Desktop/中期/NME_heatmap.png')
print("Heatmap saved")
