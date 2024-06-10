from prepare import model_param
import json
import random
import copy


RANGE=[(0,99),(9,108),(9,108),(27,126),(27,126),(39,138),(39,138),(46,145),(46,145),(53,152),(53,152),(61,160)]
STATION = ['A','B1','B2','C1','C2','D1','D2','E1','E2','F1','F2','G']

def idx(i):
    return str(i) if i in range(1,7) else '0'


num_train=16
param=model_param()
E={j:param[idx(j)]['arcs'] for j in range(1,num_train+1)}
V={j:param[idx(j)]['nodes'] for j in range(1,num_train+1)}
N=param['N']
pe={(('S',0),('A',t)):1 for t in range(100)}


value_lb = -1e8
# 选择一辆火车编号
train_idx = 9
# 获得对应节点和边
edge_name = E[train_idx]
vertex_name = V[train_idx]
value_edge = {name: random.randint(-5,-4) for name in edge_name} # 每条边的value 以(('S', 0), ('A', 2)) 等tuple作为key 

def bellman_ford(edge_name,value_edge,value_lb):
    Value = {j: {'path':[('S', 0),('A',j)], 'value': value_edge[('S', 0),('A',j)]} for j in range(100)}
    for i in range(1,len(RANGE)):
        Value_old = copy.deepcopy(Value)
        for j in range(100):
            node = tuple([STATION[i],j+RANGE[i][0]])
            matching_tuples = [item for item in edge_name if item[1] == node]
            
            Value[j]['path'] = None
            Value[j]['value'] = 2*value_lb
            if len(matching_tuples)>0:
                value_max = value_lb
                for t in matching_tuples:
                    last_node = t[0]
                    
                    value_temp = Value_old[last_node[1]-RANGE[i-1][0]]['value'] + value_edge[t]
                    
                    old_path = Value_old[last_node[1]-RANGE[i-1][0]]['path']
                    
                    if value_temp > value_max:
                        value_max = value_temp
                        Value[j]['path'] = old_path + [node]
                        Value[j]['value'] = value_max
                        
    best_value = value_lb
    best_path = None
    for j in range(100):
        if  Value[j]['value']>best_value: # 优先选择停站时间短的
            best_value = Value[j]['value']
            best_path = Value[j]['path']
    
    # 加开此列车的最优值比0小 可直接设置为此列火车不开
    if best_value < 0:
        best_value = 0
        best_path = None
    
    return best_path
    
# best_path = bellman_ford(edge_name,value_edge,value_lb)
# print(best_path)
# with open('data/shortest_path.json', 'w') as f:
#         json.dump(Value, f)

        





