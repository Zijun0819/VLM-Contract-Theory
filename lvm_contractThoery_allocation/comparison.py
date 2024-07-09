import numpy as np
from constVaribale import *
from contractTheory import Contract
from generateSite import generate_tasks_servers


def cal_avg_response_time(solution):
    return sum(task.completion_time for _, task in solution) / len(solution)


def cal_task_completion_rate(solution):
    return 100 * sum([task.completion_time < task.time_requirement for _, task in solution]) / len(solution)


def cal_difficulty_task_completion_rate(servers, solution):
    completion_l = [task.completion_time < task.time_requirement for server_id, task in solution if
                    servers[server_id].model_capabiltiy == "S"]
    completion_h = [task.completion_time < task.time_requirement for server_id, task in solution if
                    servers[server_id].model_capabiltiy == "L"]
    rate_l = 100*sum(completion_l) / len(completion_l)
    rate_h = 100*sum(completion_h) / len(completion_h)

    return rate_l, rate_h


def cal_difficulty_avg_response_time(servers, solution):
    response_time_l = [task.completion_time for server_id, task in solution if servers[server_id].model_capabiltiy == "S"]
    response_time_h = [task.completion_time for server_id, task in solution if servers[server_id].model_capabiltiy == "L"]
    avg_time_l = sum(response_time_l) / len(response_time_l)
    avg_time_h = sum(response_time_h) / len(response_time_h)

    return avg_time_l, avg_time_h


def greedy_allocation(tasks, servers):
    sorted_tasks = sorted(tasks, key=lambda x: 2 * x.task_size / x.noma_data_rate, reverse=True)
    solution = []
    for task in sorted_tasks:
        if task.target_servers is None:
            server_time = [server.cal_delta_time(task) for server in servers]
            chosen_server = servers[np.argmin(server_time)]
            chosen_server.assign_task(task)
            solution.append((chosen_server.server_id - 1, task))
        else:
            server_time = [server.cal_delta_time(task) for server in task.target_servers]
            chosen_server = task.target_servers[np.argmin(server_time)]
            chosen_server.assign_task(task)
            solution.append((chosen_server.server_id - 1, task))

    for edge_server in servers:
        sorted_edge_tasks = sorted(edge_server.assigned_tasks,
                                   key=lambda x: 2 * x.task_size / x.noma_data_rate + x.relay_time[
                                       edge_server.server_id - 1])
        for i, task in enumerate(sorted_edge_tasks):
            task.completion_time = edge_server.get_min_trans() + edge_server.time_per_img * (
                i + 1) if i == 0 else edge_server.time_per_img * (i + 1)

    f1_score = cal_task_completion_rate(solution)
    f2_score = cal_avg_response_time(solution)
    f3_score = cal_difficulty_task_completion_rate(servers, solution)
    f4_score = cal_difficulty_avg_response_time(servers, solution)

    return solution, f1_score, f2_score, f3_score, f4_score


def cal_no_contract_score(contract, tasks, servers):
    solution, f1_score, f2_score, f3_score, f4_score = greedy_allocation(tasks, servers)
    # print(f"Task completion rate is {f1_score * 100}, average time delay is {f2_score}")
    sum_u = 0
    sum_edge = 0
    sum_u_l = 0
    sum_s_l = 0
    cnt_s_l = int(SMALL_PERCENTAGE * EDGE_SERVER)
    cnt_l = 0
    for server_id, task in solution:
        server_cap = servers[server_id].model_capabiltiy
        score_col = 1 if server_cap == "S" else 2
        difficulty_level = 1 if server_cap == "S" else 2
        worker_u = contract.cal_worker_utility(difficulty_level, task.task_details[score_col])
        sum_edge += contract.cal_edge_utility(worker_u >= 0, difficulty_level)
        sum_u += worker_u
        if difficulty_level == 1:
            sum_u_l += worker_u
            sum_s_l += contract.cal_edge_utility(worker_u >= 0, difficulty_level)
            cnt_l += 1

    no_contract_worker_u = round(sum_u / len(tasks), 2)
    no_contract_server_u = round(sum_edge / len(servers), 2)
    l_w_u = round(sum_u_l / cnt_l, 2)
    l_s_u = round(sum_s_l / cnt_s_l, 2)

    h_w_u = round((sum_u - sum_u_l) / (len(tasks) - cnt_l), 2)
    h_s_u = round((sum_edge - sum_s_l) / (len(servers) - cnt_s_l), 2)

    return no_contract_worker_u, no_contract_server_u, f1_score, f2_score, (l_w_u, h_w_u), (l_s_u, h_s_u), f3_score, f4_score


def cal_contract_vlm_human_oracle_score(contract, tasks, servers, identifier):
    # identifier == 3, 4, 5 mean vlm, human perception, and oracle, respectively
    # when identifier == 3, substituting contract with the result of running complete contract bundle generation snippet
    # This means the result of vlm_complete_contract
    sum_utility = 0
    sum_server = 0
    cnt_s_l = int(SMALL_PERCENTAGE * EDGE_SERVER)
    cnt_l = 0
    sum_u_l = 0
    sum_s_l = 0
    for task in tasks:
        if task.task_details[identifier] == 1:
            task.target_servers = [x for x in servers if x.model_capabiltiy == "S"]
            worker_u = contract.cal_worker_utility(1, task.task_details[1])
            sum_server += contract.cal_edge_utility(worker_u >= 0, 1)
            sum_u_l += worker_u
            sum_s_l += contract.cal_edge_utility(worker_u >= 0, 1)
            cnt_l += 1
        else:
            task.target_servers = [x for x in servers if x.model_capabiltiy == "L"]
            worker_u = contract.cal_worker_utility(2, task.task_details[2])
            sum_server += contract.cal_edge_utility(worker_u >= 0, 2)

        sum_utility += worker_u

    _, f1_score, f2_score, f3_score, f4_score = greedy_allocation(tasks, servers)
    contract_vlm_worker_u = round(sum_utility / len(tasks), 2)
    contract_vlm_server_u = round(sum_server / len(servers), 2)
    l_w_u = round(sum_u_l / cnt_l, 2)
    l_s_u = round(sum_s_l / cnt_s_l, 2)

    h_w_u = round((sum_utility - sum_u_l) / (len(tasks) - cnt_l), 2)
    h_s_u = round((sum_server - sum_s_l) / (len(servers) - cnt_s_l), 2)

    return contract_vlm_worker_u, contract_vlm_server_u, f1_score, f2_score, (l_w_u, h_w_u), (l_s_u, h_s_u), f3_score, f4_score


def cal_contract_random_score(contract, tasks, servers):
    random_binary = np.random.randint(1, 3, size=len(tasks))

    sum_u = 0
    sum_server = 0
    cnt_s_l = int(SMALL_PERCENTAGE * EDGE_SERVER)
    cnt_l = 0
    sum_u_l = 0
    sum_s_l = 0
    for i, task in enumerate(tasks):
        if random_binary[i] == 1:
            task.target_servers = [x for x in servers if x.model_capabiltiy == "S"]
            worker_u = contract.cal_worker_utility(1, task.task_details[1])
            sum_server += contract.cal_edge_utility(worker_u >= 0, 1)
            sum_u_l += worker_u
            sum_s_l += contract.cal_edge_utility(worker_u >= 0, 1)
            cnt_l += 1
        else:
            task.target_servers = [x for x in servers if x.model_capabiltiy == "L"]
            worker_u = contract.cal_worker_utility(2, task.task_details[2])
            sum_server += contract.cal_edge_utility(worker_u >= 0, 2)

        sum_u += worker_u

    _, f1_score, f2_score, f3_score, f4_score = greedy_allocation(tasks, servers)
    random_worker_u = round(sum_u / len(tasks), 2)
    random_server_u = round(sum_server / len(servers), 2)

    l_w_u = round(sum_u_l / cnt_l, 2)
    l_s_u = round(sum_s_l / cnt_s_l, 2)

    h_w_u = round((sum_u - sum_u_l) / (len(tasks) - cnt_l), 2)
    h_s_u = round((sum_server - sum_s_l) / (len(servers) - cnt_s_l), 2)

    return random_worker_u, random_server_u, f1_score, f2_score, (l_w_u, h_w_u), (l_s_u, h_s_u), f3_score, f4_score


def vary_tasks_servers():
    time_rounds = 30
    contract = Contract()

    avg_all_res = np.zeros((4, 4))
    vary_difficulty_res = np.zeros((8, 4))
    for _ in range(time_rounds):
        tasks, servers = generate_tasks_servers()
        _ = contract.cal_contract_bundles()  # Assign the contract bundles with tradition contract theory
        no_contract_w, no_contract_s, no_contract_rate, no_contract_avg_time, n_c_w_d, n_c_s_d, n_c_rate_d, n_c_time_d = cal_no_contract_score(
            contract, tasks,
            servers)
        avg_all_res[0][0] += (no_contract_w / time_rounds)
        avg_all_res[1][0] += (no_contract_s / time_rounds)
        avg_all_res[2][0] += (no_contract_rate / time_rounds)
        avg_all_res[3][0] += (no_contract_avg_time / time_rounds)

        vary_difficulty_res[0][0] += (n_c_w_d[0] / time_rounds)
        vary_difficulty_res[1][0] += (n_c_w_d[1] / time_rounds)
        vary_difficulty_res[2][0] += (n_c_s_d[0] / time_rounds)
        vary_difficulty_res[3][0] += (n_c_s_d[1] / time_rounds)
        vary_difficulty_res[4][0] += (n_c_rate_d[0] / time_rounds)
        vary_difficulty_res[5][0] += (n_c_rate_d[1] / time_rounds)
        vary_difficulty_res[6][0] += (n_c_time_d[0] / time_rounds)
        vary_difficulty_res[7][0] += (n_c_time_d[1] / time_rounds)

        vlm_c_w, vlm_c_s, vlm_c_rate, vlm_c_avg_time, vlm_w_d, vlm_s_d, vlm_rate_d, vlm_time_d = cal_contract_vlm_human_oracle_score(
            contract, tasks, servers,
            identifier=3)
        avg_all_res[0][1] += (vlm_c_w / time_rounds)
        avg_all_res[1][1] += (vlm_c_s / time_rounds)
        avg_all_res[2][1] += (vlm_c_rate / time_rounds)
        avg_all_res[3][1] += (vlm_c_avg_time / time_rounds)

        vary_difficulty_res[0][1] += (vlm_w_d[0] / time_rounds)
        vary_difficulty_res[1][1] += (vlm_w_d[1] / time_rounds)
        vary_difficulty_res[2][1] += (vlm_s_d[0] / time_rounds)
        vary_difficulty_res[3][1] += (vlm_s_d[1] / time_rounds)
        vary_difficulty_res[4][1] += (vlm_rate_d[0] / time_rounds)
        vary_difficulty_res[5][1] += (vlm_rate_d[1] / time_rounds)
        vary_difficulty_res[6][1] += (vlm_time_d[0] / time_rounds)
        vary_difficulty_res[7][1] += (vlm_time_d[1] / time_rounds)
        human_c_w, human_c_s, human_c_rate, human_c_avg_time, human_w_d, human_s_d, human_rate_d, human_time_d = cal_contract_vlm_human_oracle_score(
            contract, tasks,
            servers,
            identifier=4)
        avg_all_res[0][2] += (human_c_w / time_rounds)
        avg_all_res[1][2] += (human_c_s / time_rounds)
        avg_all_res[2][2] += (human_c_rate / time_rounds)
        avg_all_res[3][2] += (human_c_avg_time / time_rounds)

        vary_difficulty_res[0][2] += (human_w_d[0] / time_rounds)
        vary_difficulty_res[1][2] += (human_w_d[1] / time_rounds)
        vary_difficulty_res[2][2] += (human_s_d[0] / time_rounds)
        vary_difficulty_res[3][2] += (human_s_d[1] / time_rounds)
        vary_difficulty_res[4][2] += (human_rate_d[0] / time_rounds)
        vary_difficulty_res[5][2] += (human_rate_d[1] / time_rounds)
        vary_difficulty_res[6][2] += (human_time_d[0] / time_rounds)
        vary_difficulty_res[7][2] += (human_time_d[1] / time_rounds)
        oracle_c_w, oracle_c_s, oracle_rate, oracle_avg_time, oracle_w_d, oracle_s_d, oracle_rate_d, oracle_time_d = cal_contract_vlm_human_oracle_score(
            contract, tasks,
            servers,
            identifier=5)
        avg_all_res[0][3] += (oracle_c_w / time_rounds)
        avg_all_res[1][3] += (oracle_c_s / time_rounds)
        avg_all_res[2][3] += (oracle_rate / time_rounds)
        avg_all_res[3][3] += (oracle_avg_time / time_rounds)

        vary_difficulty_res[0][3] += (oracle_w_d[0] / time_rounds)
        vary_difficulty_res[1][3] += (oracle_w_d[1] / time_rounds)
        vary_difficulty_res[2][3] += (oracle_s_d[0] / time_rounds)
        vary_difficulty_res[3][3] += (oracle_s_d[1] / time_rounds)
        vary_difficulty_res[4][3] += (oracle_rate_d[0] / time_rounds)
        vary_difficulty_res[5][3] += (oracle_rate_d[1] / time_rounds)
        vary_difficulty_res[6][3] += (oracle_time_d[0] / time_rounds)
        vary_difficulty_res[7][3] += (oracle_time_d[1] / time_rounds)

    print(
        f"Average utility of worker via no contract: vlm: human: oracle is {avg_all_res[0][0]}:{avg_all_res[0][1]}:{avg_all_res[0][2]}:{avg_all_res[0][3]}")
    print(
        f"Average utility of server via no contract: vlm: human: oracle is {avg_all_res[1][0]}:{avg_all_res[1][1]}:{avg_all_res[1][2]}:{avg_all_res[1][3]}")
    print(
        f"Average completion rate of no contract: vlm: human: oracle is {avg_all_res[2][0]}:{avg_all_res[2][1]}:{avg_all_res[2][2]}:{avg_all_res[2][3]}")
    print(
        f"Average response time of no contract: vlm: human: oracle is {avg_all_res[3][0]}:{avg_all_res[3][1]}:{avg_all_res[3][2]}:{avg_all_res[3][3]}")

    cnt_tasks = SITE_COUNT * SITE_MACHINES
    cnt_servers = EDGE_SERVER
    res_store = np.round(avg_all_res, 4)
    difficulty_res_store = np.round(vary_difficulty_res, 4).T
    # np.savetxt(f"task_{cnt_tasks}_server_{cnt_servers}.csv", res_store, delimiter=",")
    np.savetxt(f"vary_difficulty_task_{cnt_tasks}_server_{cnt_servers}.csv", difficulty_res_store, delimiter=",")


if __name__ == '__main__':
    vary_tasks_servers()
