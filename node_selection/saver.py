import torch
import os

class SequenceSaver():
    def __init__(self, save_path) -> None:
        self.save_path = save_path
        self.milp_state = None
        self.squence = []


    def save(self):
        data = []
        type = []
        cand = []
        node_id = []
        branch_label = []
        branch_action = []
        select_action = []
        select_label = []

        max_cand_length = 0
        max_data_length = 0
        for item in self.squence :
            if item["cand"] is not None :
                max_cand_length = max(max_cand_length,len(item["cand"]))
            if item["data"] is not None :
                max_data_length = max(max_data_length, len(item["data"]))

        continue_branch = False # dont know why for setcover problem, select node1 3 times and may branch on node1 2 times with diff var
        for item in self.squence:
            node_id.append(-1)
            branch_label.append(-1)
            branch_action.append(-1)
            select_action.append(-1)
            select_label.append(-1)
            if item['type'] == 'select':
                select_node_number = item["data"][0]
                label_node_number = item["label"][0]
                if select_node_number != 1:
                # find node feature, use node feature when select node, except for node1
                    index = node_id.index(select_node_number)
                    data.append(data[index])
                else:
                    data.append(item["data"] + [0] * (max_data_length - len(item["data"])))

                select_action[-1] = select_node_number
                select_label[-1] = label_node_number
                type.append(1)
                cand.append(item["cand"] + [-1] * (max_cand_length - len(item["cand"])))
            elif item['type'] == 'branch':
                if len(data)!=0 and type[-1] ==2 :
                    continue_branch = True
                    print(f'why continue branch on {self.save_path}')
                data.append(item["data"] + [0] * (max_data_length - len(item["data"])))
                type.append(2)
                cand.append(item["cand"] + [-1] * (max_cand_length - len(item["cand"])))
                branch_label[-1] = item["branch_label"]
                branch_action[-1] = item["data"][0]
            elif item['type'] == 'node':
                data.append(item["data"] + [0] * (max_data_length - len(item["data"])))
                type.append(3)
                cand.append([-1] * max_cand_length)
                node_id[-1] = item["node_number"]

        data_tensor = torch.tensor(data, dtype=torch.float32)
        type_tensor = torch.tensor(type, dtype=torch.int64)
        cand_tensor = torch.tensor(cand, dtype=torch.int64)
        node_id_tensor = torch.tensor(node_id, dtype=torch.int64)
        branch_label_tensor = torch.tensor(branch_label, dtype=torch.int64)
        branch_action_tensor = torch.tensor(branch_action, dtype=torch.int64)
        select_action_tensor = torch.tensor(select_action, dtype=torch.int64)
        select_label_tensor = torch.tensor(select_label, dtype=torch.int64)
        if len(node_id)>=5 and not continue_branch:
            if not os.path.exists(self.save_path) :
                os.makedirs(self.save_path , exist_ok=True)
            torch.save(data_tensor, self.save_path + '/data.pt')
            torch.save(type_tensor, self.save_path + '/type.pt')
            torch.save(cand_tensor, self.save_path + '/cand.pt')
            torch.save(node_id_tensor, self.save_path + '/node_id.pt')
            torch.save(branch_label_tensor, self.save_path + '/branch_label.pt')
            torch.save(branch_action_tensor, self.save_path + '/branch_action.pt')
            torch.save(select_action_tensor, self.save_path + '/select_action.pt')
            torch.save(select_label_tensor, self.save_path + '/select_label.pt')
            torch.save(self.milp_state, self.save_path + '/state.pt')
            print(f"Saved sequence to {self.save_path}")
        else:
            print(f"Not saved sequence to {self.save_path} because of too short")


