import json
import os
from datetime import datetime

class TrainingLogger:
    """
    简单的训练日志记录器，将指标保存为JSON格式
    """
    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志文件名（包含时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.json")
        
        # 初始化日志数据
        self.log_data = {
            "metadata": {
                "start_time": timestamp,
                "log_file": self.log_file
            },
            "epochs": [],
            "steps": []
        }
        
        print(f"Training logger initialized. Log file: {self.log_file}")
    
    def log_step(self, epoch, step, metrics):
        """
        记录每个step的指标
        metrics: dict, 包含各种指标
        """
        step_data = {
            "epoch": epoch,
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        self.log_data["steps"].append(step_data)
    
    def log_epoch(self, epoch, metrics):
        """
        记录每个epoch的指标
        metrics: dict, 包含各种指标
        """
        epoch_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        self.log_data["epochs"].append(epoch_data)
        
        # 实时保存到文件
        self.save()
    
    def save(self):
        """保存日志到文件"""
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def get_metrics_for_plotting(self):
        """获取用于绘图的指标数据"""
        epochs = self.log_data["epochs"]
        steps = self.log_data["steps"]
        
        # 提取epoch级别的指标
        epoch_metrics = {}
        if epochs:
            for key in epochs[0].keys():
                if key not in ["epoch", "timestamp"]:
                    epoch_metrics[key] = [epoch[key] for epoch in epochs]
            epoch_metrics["epoch"] = [epoch["epoch"] for epoch in epochs]
        
        # 提取step级别的指标
        step_metrics = {}
        if steps:
            for key in steps[0].keys():
                if key not in ["epoch", "step", "timestamp"]:
                    step_metrics[key] = [step[key] for step in steps]
            step_metrics["global_step"] = [
                step["epoch"] * 1000 + step["step"] for step in steps  # 假设每个epoch有1000步
            ]
        
        return epoch_metrics, step_metrics

def plot_training_curves(log_file, save_path="./plots"):
    """
    绘制训练曲线
    """
    import matplotlib.pyplot as plt
    import json
    import os
    
    os.makedirs(save_path, exist_ok=True)
    
    # 加载日志数据
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    epochs = data["epochs"]
    if not epochs:
        print("No epoch data found!")
        return
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves', fontsize=16)
    
    # 提取数据
    epoch_nums = [e["epoch"] for e in epochs]
    
    # 1. Loss曲线
    if "select_loss" in epochs[0] and "branch_loss" in epochs[0]:
        select_losses = [e["select_loss"] for e in epochs]
        branch_losses = [e["branch_loss"] for e in epochs]
        total_losses = [e.get("total_loss", s + b) for e, s, b in zip(epochs, select_losses, branch_losses)]
        
        axes[0, 0].plot(epoch_nums, select_losses, label='Select Loss', color='blue')
        axes[0, 0].plot(epoch_nums, branch_losses, label='Branch Loss', color='red')
        axes[0, 0].plot(epoch_nums, total_losses, label='Total Loss', color='green')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # 2. Accuracy曲线
    if "select_acc" in epochs[0] and "branch_acc" in epochs[0]:
        select_accs = [e["select_acc"] for e in epochs]
        branch_accs = [e["branch_acc"] for e in epochs]
        
        axes[0, 1].plot(epoch_nums, select_accs, label='Select Accuracy', color='blue')
        axes[0, 1].plot(epoch_nums, branch_accs, label='Branch Accuracy', color='red')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # 3. Steps曲线
    if "select_steps" in epochs[0] and "branch_steps" in epochs[0]:
        select_steps = [e["select_steps"] for e in epochs]
        branch_steps = [e["branch_steps"] for e in epochs]
        
        axes[1, 0].plot(epoch_nums, select_steps, label='Select Steps', color='blue')
        axes[1, 0].plot(epoch_nums, branch_steps, label='Branch Steps', color='red')
        axes[1, 0].set_title('Steps per Epoch')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # 4. Time曲线
    if "duration" in epochs[0]:
        durations = [e["duration"] for e in epochs]
        
        axes[1, 1].plot(epoch_nums, durations, label='Epoch Duration', color='purple')
        axes[1, 1].set_title('Training Time per Epoch')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_file = os.path.join(save_path, "training_curves.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training curves saved to: {plot_file}")

if __name__ == "__main__":
    # 示例用法
    logger = TrainingLogger()
    
    # 模拟训练过程
    for epoch in range(5):
        for step in range(10):
            # 记录step指标
            step_metrics = {
                "select_loss": 1.0 - epoch * 0.1 + step * 0.01,
                "branch_loss": 0.8 - epoch * 0.08 + step * 0.005,
                "select_acc": 0.5 + epoch * 0.1 + step * 0.01,
                "branch_acc": 0.6 + epoch * 0.08 + step * 0.005
            }
            logger.log_step(epoch, step, step_metrics)
        
        # 记录epoch指标
        epoch_metrics = {
            "select_loss": 1.0 - epoch * 0.1,
            "branch_loss": 0.8 - epoch * 0.08,
            "select_acc": 0.5 + epoch * 0.1,
            "branch_acc": 0.6 + epoch * 0.08,
            "select_steps": 10,
            "branch_steps": 8,
            "duration": 30.0 + epoch * 2.0
        }
        logger.log_epoch(epoch, epoch_metrics)
    
    # 绘制曲线
    plot_training_curves(logger.log_file) 