import torch
import os

class SequenceSaver():
    def __init__(self, save_path) -> None:
        self.save_path = save_path
        self.milp_state = None
        self.squence = []
        if not os.path.exists(self.save_path) :
            os.makedirs(self.save_path , exist_ok=True)

    def save(self):
        data = []
        type = []
        cand = []
        node_id = []

        max_cand_length = 0
        max_data_length = 0
        for item in self.squence :
            if item["cand"] is not None :
                max_cand_length = max(max_cand_length,len(item["cand"]))
            if item["data"] is not None :
                max_data_length = max(max_data_length, len(item["data"]))
        for item in self.squence:
            node_id.append(-1)
            if item['type'] == 'select':
                data.append(item["data"] + [0] * (max_data_length - len(item["data"])))
                type.append(1)
                cand.append(item["cand"] + [-1] * (max_cand_length - len(item["cand"])))
            elif item['type'] == 'branch':
                data.append(item["data"] + [0] * (max_data_length - len(item["data"])))
                type.append(2)
                cand.append(item["cand"] + [-1] * (max_cand_length - len(item["cand"])))
            elif item['type'] == 'node':
                data.append(item["data"] + [0] * (max_data_length - len(item["data"])))
                type.append(3)
                cand.append([-1] * max_cand_length)
                node_id[-1] = item["node_number"]

        data_tensor = torch.tensor(data, dtype=torch.float32)
        type_tensor = torch.tensor(type, dtype=torch.int64)
        cand_tensor = torch.tensor(cand, dtype=torch.int64)
        node_id_tensor = torch.tensor(node_id, dtype=torch.int64)

        torch.save(data_tensor, self.save_path + '/data.pt')
        torch.save(type_tensor, self.save_path + '/type.pt')
        torch.save(cand_tensor, self.save_path + '/cand.pt')
        torch.save(node_id_tensor, self.save_path + '/node_id.pt')
        torch.save(self.milp_state, self.save_path + '/state.pt')

        print(f"Saved sequence to {self.save_path}")
