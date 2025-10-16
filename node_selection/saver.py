import torch
import os
import datetime

class SequenceSaver():
    def __init__(self, save_path) -> None:
        # 添加时间戳到保存路径
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.save_path = f"{save_path}_{timestamp}"
        self.milp_state = None
        self.squence = []


    def save(self):
        if not os.path.exists(self.save_path) :
            os.makedirs(self.save_path , exist_ok=True)
        torch.save(self.squence, self.save_path + '/sequence.pt')



