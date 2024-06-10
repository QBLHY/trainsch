'''
用于检查解是否为有效解
'''


from prepare import model_param
import json
from bellman_ford_alg import bellman_ford
import copy
import random

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

with open(f'./data/shortest_path_change.json', 'r') as f:
    sol_arcs = json.load(f)

# for train_idx_replace in range(1,num_train+1):
    
#     # 获得对应节点和边
#     edge_name = E[train_idx_replace]
#     vertex_name = V[train_idx_replace]
#     value_edge = {name: 0 for name in edge_name}

# 需要统计一遍每个节点的Y
conflicting_node =[]
X = {j:{} for j in range(1,num_train+1)}
for j in range(1,num_train+1):
    for name in E[j]:
        X[j][name] = 0
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
    sum_y = 0
    for node in N[str(name)]:
        sum_y += Y[node]
    if sum_y >1:
        conflicting_node.append((name,sum_y))
if len(conflicting_node)>0:
    print(conflicting_node)