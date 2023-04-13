import numpy as np
from matplotlib import pyplot as plt


epochs = range(1, 13)
baseline = [35.8, 40.7, 42.7, 44.3, 45.4, 45.6, 45.3, 46.2, 49.8, 50.4, 50.7, 50.8]
ours = [38.9, 42.4, 43.8, 45.3, 45.7, 46.8, 46.3, 47.1, 50.0, 50.4, 50.9, 50.9]

# plt.axhline(y=37.9, c='gray', linestyle='--')

plt.plot(epochs, baseline, c='blue', marker='o', markersize=6, label='Random Initialization')
plt.plot(epochs, ours, c='red', marker='*', markersize=10, label='Ours')


x_ticks = np.linspace(1, 12, 12)
plt.xticks(x_ticks, fontsize=9)

# plt.xlim(0.5, 12.5)
# plt.ylim(6, 40)

plt.xlabel('Fine-tuning Epoch')
plt.ylabel('mAP')

plt.grid(linewidth=0.2)
plt.legend()

plt.tight_layout()
plt.savefig('test.png', dpi=512)