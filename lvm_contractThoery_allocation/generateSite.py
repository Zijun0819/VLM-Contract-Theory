import math
import os.path
import random

import numpy as np
import pandas as pd

from constVaribale import *


class Task(object):     # utility model should take time, task score, and charged money into consideration
    def __init__(self, *, task_id, task_details, task_size):
        self.task_id = task_id
        self.time_requirement = 1000    # Unit is ms
        self.completion_time = None
        self.task_decision = None
        self.target_servers = None    # After contract theory selection, target offloading servers
        self.task_size = task_size      # The unit is Mb
        self.noma_data_rate = np.random.randint(NOMA_MIN, NOMA_MAX)     # The unit is Mb
        # The latency from smart gate to each server via wired cable, the unit is ms
        self.relay_time = np.random.randint(1, 5, size=EDGE_SERVER) * np.random.randint(RELAY_MIN,
                                                                                                  RELAY_MAX + 1,
                                                                                                  size=EDGE_SERVER)
        self.task_score = None
        self.task_target_score_1 = 1.3
        self.task_target_score_2 = 1.4
        self.task_cost = None
        self.task_details = task_details


# utility model should take data collection, model training, and task execution into consideration
class EdgeServer(object):
    def __init__(self, server_id, model_capability):
        self.server_id = server_id
        self.model_capabiltiy = model_capability    # L or S, represent the large model or small model is deployed
        self.computing_capability = random.randint(3, 5)
        # Simulation results on RTX 4090 demonstrate this parameter
        self.time_per_img = round(random.uniform(117.37, 283.26) / self.computing_capability, 2)
        self.assigned_tasks = []

    def assign_task(self, task: Task):
        self.assigned_tasks.append(task)

    def clear_tasks(self):
        self.assigned_tasks = []

    def get_min_trans(self):
        if len(self.assigned_tasks) != 0:
            trans_time = [2000 * x.task_size / x.noma_data_rate + x.relay_time[self.server_id - 1] for x in
                      self.assigned_tasks]
        else:
            trans_time = [1e-3]

        return min(trans_time)

    def queuing_time(self):
        return len(self.assigned_tasks)*self.time_per_img + self.get_min_trans()

    def cal_delta_time(self, task):
        self.assigned_tasks.append(task)
        after_time = self.queuing_time()
        self.assigned_tasks.pop()

        return after_time


def generate_tasks_servers() -> tuple:
    image_dir = "Images_pool"

    edge_servers = list()
    tasks = list()
    task_id = 1
    server_id = 1
    df = pd.read_excel("difficulty.xlsx")
    tasks_pool = df.iloc[:, :].to_numpy()

    servers = np.arange(1, EDGE_SERVER+1)
    # np.random.shuffle(servers)
    servers_large = servers > int(SMALL_PERCENTAGE*EDGE_SERVER)

    # Simulate the construction site
    for i in range(SITE_COUNT*SITE_MACHINES):
        # difficulty_level = 1 if i < math.floor(SITE_MACHINES*SITE_COUNT*CONTRACT_PERCENTAGE) else 2
        selected_task = np.random.randint(0, tasks_pool.shape[0])
        task_details = tasks_pool[selected_task, :]
        task_size = os.path.getsize(os.path.join(image_dir, f"{task_details[0]}.png")) / (1024*1024)
        tasks.append(Task(task_id=task_id, task_details=task_details, task_size=task_size))
        task_id = task_id + 1

    # Simulate the edge server with cell that radius is equal to 500 meter
    for _ in range(EDGE_SERVER):
        model_capability = "L" if servers_large[server_id-1] else "S"
        edge_servers.append(EdgeServer(server_id=server_id, model_capability=model_capability))
        server_id = server_id + 1

    return tasks, edge_servers


if __name__ == '__main__':
    # random.seed(RANDOM_SEED)
    # np.random.seed(RANDOM_SEED)
    tasks, edge_servers = generate_tasks_servers()
    print("===============")
