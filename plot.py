import matplotlib.pyplot as plt
import numpy as np

def smooth(values, weight=0.9):
    """简单滑动平均"""
    smoothed = []
    last = values[0]
    for val in values:
        smoothed_val = last * weight + (1 - weight) * val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# 读取日志文件
log_path = '51log.txt'
epochs, losses = [], []

with open(log_path, 'r') as f:
    for line in f:
        if 'Epoch' in line and 'Loss' in line:
            parts = line.strip().split()
            try:
                epoch = int(parts[1][:-1])
                loss = float(parts[4])
                epochs.append(epoch)
                losses.append(loss)
            except (IndexError, ValueError):
                continue

# 平滑处理
losses_smoothed = smooth(losses, weight=0.9)

# 画图
plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, color='lightgray', label='Original Loss', linestyle='--')
plt.plot(epochs, losses_smoothed, color='blue', label='Smoothed Loss', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Foundation model training')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('smoothed_loss.png')
plt.show()
