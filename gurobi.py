import gurobipy as gp
from gurobipy import GRB
from prepare import model_param
import json


def idx(i):
    return str(i) if i in range(1,7) else '0'

num_train=16
param=model_param()
E={j:param[idx(j)]['arcs'] for j in range(1,num_train+1)}
V={j:param[idx(j)]['nodes'] for j in range(1,num_train+1)}
dm={j:param[idx(j)]['delta_minus'] for j in range(1,num_train+1)} 
dp={j:param[idx(j)]['delta_plus'] for j in range(1,num_train+1)}
N=param['N']
pe={(('S',0),('A',t)):1 for t in range(100)}

# Create a new model
m = gp.Model("train_schedule") 

# create vars
trains = {j: m.addVars(E[j], obj=pe,vtype=GRB.BINARY, name=f"xe_{j}") for j in range(1, num_train+1)}
z = {j: m.addVars(V[j],  name=f"zjv_{j}") for j in range(1, num_train+1)}
y = m.addVars(param['nodes'], name="yv")

# add constraints
# constraint (2): at most one starting arc for each train
for i in range(1,num_train+1):
    starts=dp[i][str(('S',0))]
    m.addConstr(gp.quicksum(trains[i][arc] for arc in starts)<=1,name=f"start_{i}")


# constraint (3): Non-artiﬁcial  nodes  must  have  equal  in  and  out  degrees
for i in range(1,num_train+1):
    for v in V[i]:
        dmv=dm[i][str(v)] # delta_minus_i(v)
        dpv=dp[i][str(v)] # delta_plus_i(v)
        m.addConstr(gp.quicksum(trains[i][arc] for arc in dmv)==gp.quicksum(trains[i][arc] for arc in dpv),name=f"degree_{i}_{v}")

# constraint (4): at most one ending arc for each train
for i in range(1,num_train+1):
    ends=dm[i][str(('T',0))]
    m.addConstr(gp.quicksum(trains[i][arc] for arc in ends)<=1,name=f"end_{i}")

# constraint (5): relation of z and x_e
for i in range(1,num_train+1):
    for v in V[i]:
        dmv=dm[i][str(v)]
        m.addConstr(z[i][v]==gp.quicksum(trains[i][arc] for arc in dmv),name=f"z_{i}_{v}")

# constraint (6): relation of y and z
for v in param['nodes']:
    m.addConstr(y[v]==gp.quicksum(z[i][v] for i in range(1,num_train+1)),name=f"y_{v}")


#constraint (7): headaway constraints
for node in param['nodes']:
    conflict=N[str(node)]
    if len(conflict)>7: print('error in conflict node', node)
    m.addConstr(gp.quicksum(y[v] for v in conflict)<=1,name=f"headaway_{node}")



# 设置目标函数
m.modelSense=GRB.MAXIMIZE


print('start optimization')
m.optimize()

# 输出结果并保存
sol_arcs={}
if m.Status == GRB.OPTIMAL:
    for i in range(1,num_train+1):
        sol_arcs[i]=[]
        for arc in E[i]:
            if trains[i][arc].x>0.5:
                sol_arcs[i].append(arc[0])
                print('train',i,'arc',arc)
        print('-----------------------')

with open('data/gb_solution.json', 'w') as f:
    json.dump(sol_arcs, f)



