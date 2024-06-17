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

with open(f'./data/shortest_path_lagrangian_74_multiplier.json', 'r') as f:
    lagrangian_multiplier = json.load(f)
with open(f'./data/shortest_path_lagrangian_74.json', 'r') as f:
    sol_arcs = json.load(f)

start=time.time()
dual_value = {}
for train_idx in range(1,num_train+1):
    path = sol_arcs[str(train_idx)]

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
    dual = 0
    for j in range(len(path)-1):
        name = tuple([tuple(path[j]),tuple(path[j+1])])
        dual += value_edge[name]
    dual_value[str(train_idx)]=dual


sorted_dual = sorted(dual_value.items(), key=lambda x: x[1],reverse=False)

print(sorted_dual)

train_list = [i[0] for i in sorted_dual]
while len(train_list)>0:
    train_idx_replace = int(train_list.pop(0))
    # 获得对应节点和边
    edge_name = E[train_idx_replace]
    vertex_name = V[train_idx_replace]
    value_edge = {name: 0 for name in edge_name}

    # 需要重新统计一遍除去现有的一列火车后每个节点的Y
    conflicting_node =[]
    X = {j:{} for j in range(1,num_train+1)}
    for j in range(1,num_train+1):
        for name in E[j]:
            X[j][name] = 0
        if j != train_idx_replace: # 除去现有的一列火车
            path = sol_arcs[str(j)]
            if path: 
                for k in range(len(path)-1):
                    name = tuple([tuple(path[k]),tuple(path[k+1])])
                    X[j][name] = 1

    Z = {j:{} for j in range(1,num_train+1)}
    for train_idx in range(1,num_train+1):
        for name in V[1]:
            Z[train_idx][name] = sum([ X[train_idx][neighbor_name] for neighbor_name in dm[train_idx][str(name)]])
    Y = {}
    for name in V[1]:
        Y[name] = sum([Z[train_idx][name] for train_idx in range(1,num_train+1)])

    for name in V[1]:
        for node in N[str(name)]:
            sum_y = 0
            for k in N[str(node)]:
                sum_y += Y[k]
            if sum_y >0:
                conflicting_node.append(name)
                break

    matching_edges = []
    for node in conflicting_node:
        matching_edges += [item for item in edge_name if item[1] == node]

    
    for name in matching_edges:
        value_edge[name] = -1000 # 给冲突节点包含的边较大惩罚，相当于删去此边
    
    best_path_temp = bellman_ford(edge_name=edge_name,value_edge = value_edge,value_lb=-1e8)
    sol_arcs[str(train_idx_replace)] = best_path_temp

    print(train_idx_replace)

    if best_path_temp == None:
        train_list.append(train_idx_replace)
    
end=time.time()
print('Running time of heuristic algorithm is {:.2f}'.format(end-start))
with open(f'./data/shortest_path_change.json', 'w') as f:
    json.dump(sol_arcs, f)







    


    


