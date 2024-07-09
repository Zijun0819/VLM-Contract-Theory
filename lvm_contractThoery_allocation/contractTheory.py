import math
import csv
import pandas as pd
from generateSite import generate_tasks_servers
import numpy as np


class Contract(object):
    def __init__(self):
        self.I_r1 = 1.3
        self.I_r2 = 1.4
        self.theta_L = 1
        self.theta_H = math.sqrt(2)
        self.eta_1 = 5
        self.eta_2 = 2.5e2
        self.eta_3 = 1
        self.beta_L = 0.4
        self.beta_H = 1 - self.beta_L

        self.I_L = None
        self.I_H = None
        self.p_L = None
        self.p_H = None

    def worker_utility(self, theta, I):
        return theta * np.log(self.eta_2 * (I - self.I_r1)) + self.eta_3 * (I - self.I_r2)

    def cal_contract_bundles(self):
        contract_bundles = list()

        self.I_L = self.I_r1 + (self.theta_L - self.beta_H * self.theta_H) / (self.beta_L * (self.eta_1 - self.eta_3))
        self.I_H = self.I_r1 + (self.theta_H / (self.eta_1 - self.eta_3))
        self.p_L = self.theta_L * np.log(self.eta_2 * (self.I_L - self.I_r1)) + self.eta_3 * (self.I_L - self.I_r2)
        self.p_H = self.p_L + self.worker_utility(self.theta_H, self.I_H) - self.worker_utility(self.theta_H, self.I_L)

        contract_bundles.append((round(self.p_L, 3), self.I_L))
        contract_bundles.append((round(self.p_H, 3), self.I_H))

        return contract_bundles

    def cal_complete_contract_bundles(self):
        contract_bundles = list()

        self.I_L = self.I_r1 + self.theta_L / (self.eta_1 - self.eta_3)
        self.I_H = self.I_r1 + self.theta_H / (self.eta_1 - self.eta_3)
        self.p_L = self.theta_L * np.log(self.eta_2 * (self.I_L - self.I_r1)) + self.eta_3 * (self.I_L - self.I_r2)
        self.p_H = self.theta_H * np.log(self.eta_2 * (self.I_H - self.I_r1)) + self.eta_3 * (self.I_H - self.I_r2)

        contract_bundles.append((round(self.p_L, 3), self.I_L))
        contract_bundles.append((round(self.p_H, 3), self.I_H))

        return contract_bundles

    def cal_worker_utility(self, difficulty, score):
        if difficulty == 1:
            return self.theta_L * np.log(self.eta_2 * (score - self.I_r1)) + self.eta_3 * (score - self.I_r2) - self.p_L
        else:
            return self.theta_H * np.log(self.eta_2 * (score - self.I_r1)) + self.eta_3 * (score - self.I_r2) - self.p_H

    def cal_theta_worker_utility(self, theta, model_capabiltiy, score):
        if model_capabiltiy == "S":
            return theta * np.log(self.eta_2 * (score - self.I_r1)) + self.eta_3 * (score - self.I_r2) - self.p_L
        else:
            return theta * np.log(self.eta_2 * (score - self.I_r1)) + self.eta_3 * (score - self.I_r2) - self.p_H

    def cal_oracle_worker_utility(self, difficulty, score):
        if difficulty == 1:
            return np.log(self.eta_2 * (score - self.I_r1)) + self.eta_3 * (score - self.I_r2) - self.p_L
        else:
            return np.log(self.eta_2 * (score - self.I_r1)) + self.eta_3 * (score - self.I_r2) - self.p_H

    def cal_edge_utility(self, satisfy, difficulty):
        m_fee = 10
        if satisfy:
            price_L = self.p_L
            price_H = self.p_H
        else:
            price_L = 0
            price_H = 0
        return price_L - self.eta_1 * self.I_L + m_fee if difficulty == 1 else price_H - self.eta_1 * self.I_H + m_fee


def contract_test():
    contract = Contract()
    contracts = contract.cal_contract_bundles()
    print(contracts)
    # complete_contracts = contract.cal_complete_contract_bundles()
    # print(complete_contracts)
    tasks, edge_servers = generate_tasks_servers()
    data_length = len(tasks)
    random_binary = np.random.randint(1, 3, size=data_length)
    utility_L = contract.cal_worker_utility(1, 1.44)
    utility_H = contract.cal_worker_utility(2, 1.54)
    print(f"Utility of low type is {utility_L}, high type is {utility_H}")
    for i in range(3):
        sum_utility = 0
        for task in tasks:
            if task.task_details[3 + i] == 1:
                worker_u = contract.cal_worker_utility(1, task.task_details[1])
                # print(f"Utility of task {task.task_details[0]} is: {worker_u}, difficulty level is 1")
            else:
                worker_u = contract.cal_worker_utility(2, task.task_details[2])
                # print(f"Utility of task {task.task_details[0]} is: {worker_u}, difficulty level is 2")

            sum_utility += worker_u
        print(f"Average utility for lvm{i + 1} is {round(sum_utility / len(tasks), 2)}")

    sum_utility = 0
    for i in range(data_length):
        worker_u = contract.cal_worker_utility(random_binary[i], tasks[i].task_details[random_binary[i]])
        sum_utility += worker_u

    print(f"Average utility for random is {round(sum_utility / data_length, 2)}")



def oracle_difficulty():
    df = pd.read_excel("difficulty.xlsx")
    tasks_pool = df.iloc[:, :].to_numpy()

    contract = Contract()
    _ = contract.cal_contract_bundles()
    oracle_difficulty_list = []

    # columns: (image_id, small, large, lvm_1, lvm_2, lvm_3, human)
    for i in range(tasks_pool.shape[0]):
        utility_w_1 = contract.cal_worker_utility(1, tasks_pool[i][1])
        utility_w_2 = contract.cal_worker_utility(2, tasks_pool[i][2])
        difficulty_level = 1 if utility_w_1 > utility_w_2 else 2
        oracle_difficulty_list.append(difficulty_level)

    with open('oracle_difficulty.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for item in oracle_difficulty_list:
            writer.writerow([item])

    print("=========Done!=========")


if __name__ == '__main__':
    oracle_difficulty()
    contract_test()
