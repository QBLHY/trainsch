from prepare import model_param
import json
from bellman_ford_alg import bellman_ford
import copy
import random
import time

RANGE=[(0,99),(9,108),(9,108),(27,126),(27,126),(39,138),(39,138),(46,145),(46,145),(53,152),(53,152),(61,160)]

STATION = ['A','B1','B2','C1','C2','D1','D2','E1','E2','F1','F2','G']
def idx(i):
    return str(i) if i in range(1,7) else '0'

num_train=16
param=model_param()
E={j:param[idx(j)]['arcs'] for j in range(1,num_train+1)}
V={j:param[idx(j)]['nodes'] for j in range(1,num_train+1)}
dm={j:param[idx(j)]['delta_minus'] for j in range(1,num_train+1)}
N=param['N']

eta = 0.1
rho = 0.95
lagrangian_multiplier = {str(name): random.uniform(0.01,0.2) for name in V[1]}
total_epoch = 80

start_time = time.time()
for epoch in range(total_epoch):
    eta = eta * rho
    use_train_num = 0
    X = {j:{} for j in range(1,num_train+1)}
    for j in range(1,num_train+1):
        for name in E[j]:
            X[j][name] = 0
    path = {j:{} for j in range(1,num_train+1)}
    for train_idx in range(1,num_train+1):
        
        # 获得对应节点和边
        edge_name = E[train_idx]
        vertex_name = V[train_idx]

        value_edge = {}
        for name in edge_name:
            weight_sum = 0
            if name[1][0] != 'T' and name[1] in V[train_idx]:
                Neighbor = N[str(name[1])]
                for n in Neighbor:
                    weight_sum += lagrangian_multiplier[str(n)]
                if name[0][0] != 'S':
                    value_edge[name] = -weight_sum
                else:
                    value_edge[name] = 1
        
        best_path_temp = bellman_ford(edge_name=edge_name,value_edge = value_edge,value_lb=-1e8)
        path[train_idx] = best_path_temp

        if best_path_temp: # 如果最优路径不为0 更新路径 增加使用车数
            use_train_num += 1
            for j in range(len(best_path_temp)-1):
                name = tuple([best_path_temp[j],best_path_temp[j+1]])
                X[train_idx][name] = 1
    
   # 保存60步后使用了16辆车的解（解肯定为无效解）
    if (epoch+1) >60 and use_train_num == 16:
        with open(f'data/shortest_path_lagrangian_{epoch+1}.json', 'w') as f:
            json.dump(path, f)
        with open(f'data/shortest_path_lagrangian_{epoch+1}_multiplier.json', 'w') as f:
            json.dump(lagrangian_multiplier, f)

    # 更新乘子，判断解是否有效，计算上界
    valid_flag = 1
    final_value = 0
    Z = {j:{} for j in range(1,num_train+1)}
    for train_idx in range(1,num_train+1):
        for name in V[1]:
            Z[train_idx][name] = sum([ X[train_idx][neighbor_name] for neighbor_name in dm[train_idx][str(name)]])
    Y = {}
    for name in V[1]:
        Y[name] = sum([Z[train_idx][name] for train_idx in range(1,num_train+1)])
    for name in V[1]:
        sum_y = -1
        for node in N[str(name)]:
            sum_y += Y[node]
        

        # 计算结果
        final_value += -lagrangian_multiplier[str(name)]* sum_y
        lagrangian_multiplier[str(name)] = max(0, lagrangian_multiplier[str(name)]+eta * sum_y)
        
        if sum_y>0:
            valid_flag = 0
    final_value +=  use_train_num

    print(f'epoch: {epoch+1} target value: {final_value:.2f} valid: {valid_flag} use train number:{use_train_num}')
end_time = time.time()
cost_time = end_time-start_time
print(cost_time)

    #   break
        
            

    
        


        
        
        


