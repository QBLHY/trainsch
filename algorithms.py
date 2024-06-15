import copy
from tqdm import tqdm
import time 
from prepare import conflict_node

###########################################
### 这个文件夹中包含了一些需要被调用的算法
###########################################
def idx(i):
    return str(i) if i in range(1,7) else '0'


'''
Bellman-Ford Algorithm for !!LONGEST!! path problem only for the space-time network model.
INPUT:
    nodes: list/set of nodes (no artificial nodes)
    arcs: list/set of arcs (include arcs from artificial nodes to real nodes and from real nodes to artificial nodes)
    weights: dictionary of weights of arcs
    start: start node
    end: end node
OUTPUT:
    dist: the longest path from start to end 
    path: the longest path from start to end (as a list), 列表中没有起点，但包含终点. 如果路径不存在，则返回空列表

WARNING: We assume there is no positive cycle in the graph!!!
        The algorithm is only for the space-time network model.
'''
def BF(nodes,arcs,weights,start,end):
    dist = {node: float('-inf') for node in nodes}
    # 补充artifactial nodes
    dist[('S',0)]=float('-inf')
    dist[('T',0)]=float('-inf')
    dist[start] = 0 
    path = {node: [] for node in nodes}
    # 补充artifactial nodes
    path[('S',0)]=[]
    path[('T',0)]=[]
    for _ in range(13): # 实际上space-time network中最长的路径长度为13
        for arc in arcs:
            if dist[arc[1]] < dist[arc[0]] + weights[arc]:
                dist[arc[1]] = dist[arc[0]] + weights[arc]
                path[arc[1]] = path[arc[0]] + [arc[1]]
    return dist[end], path[end]

'''
将路径转换为字典
INPUT
    path: a dictionary, key is the train NO, value is a list of path 包含虚拟的终点，不包含虚拟的起点
OUTPUT
    x: a dictionary, key is the train NO, value is a dictionary.
'''
def path_to_vec(path,param):
    x={j:{arc:0 for arc in param[idx(j)]['arcs']}   for j in path.keys()}
    for train in path.keys():
        if len(path[train])==0 : continue
        x[train][(('S',0),tuple(path[train][0]))]=1
        for _ in range(len(path[train])-1):
            x[train][(path[train][_],path[train][_+1])]=1
    return x

'''
将字典转换为路径
INPUT
    x: a dictionary, key is the train NO, value is a dictionary.
OUTPUT
    path: a dictionary, key is the train NO, value is a list of path 包含虚拟的终点，不包含虚拟的起点
'''
def vec_to_path(x):
    path={}
    for train in x.keys():
        path[train]=[]
        for arc in x[train].keys():
            if x[train][arc]==1:
                path[train].append(arc[1])
    return path

'''
Heuristic Algorithm 
INPUT:
param: 包含了G,E, G^j,E^j,N,delta,pe ...space-time network 的模型参数
mul: 拉格朗日乘子的值  a dictionary , key = V
sol: a schedule of 16 trains (maybe not feasible), 默认sol 中的路径若不为空，包含了虚拟的终点,不包括虚拟的起点
num_train: 默认为16

OUTPUT: 成功则返回一个可行解fea_sol, 否则返回NONE. fea_sol也是一些路径
'''
def find_feasible(param,mul,sol,num_train=16):
    fea_sol=copy.deepcopy(sol)
    # compute dual obj function for each train
    dual_obj = {train:0 for train in range(1,num_train+1)}
    pe=param['pe']
    # train 为列车编号
    for train in range(1,num_train+1): 
        path = sol[train]
        # 获得对应节点和边
        for _ in range(len(path)-1):
            arc= tuple([tuple(path[_]),tuple(path[_+1])])
            t=arc[1] # arc=(*,t)
            # 计算这条边arc在dual obj function中前面对应的系数
            coeff=pe[arc]
            if t[0]!='T':
                neighbors=param['N'][str(t)]
                for v in neighbors:
                    coeff-=mul[v]
            dual_obj[train]+=coeff
    
    # 对dual_obj 进行排序
    sorted_dual = sorted(dual_obj.items(), key=lambda x: x[1],reverse=False) 
    priority_list=[item[0] for item in sorted_dual]

    # 逐个替换列车
    for _ in range(1): # 最多执行1轮
        for train in priority_list:
            # 获得对应节点和边
            E_j = set(param[idx(train)]['arcs'])
            V_j = set(param[idx(train)]['nodes'])
            # 去掉与其他列车冲突的node以及arc
            for t in range(1,num_train+1):
                if t!=train:
                    path = fea_sol[t] # the path of the train t of the current fea_sol
                    if (len(path)==0): continue
                    # nodes on this path
                    nodes={tuple(path[_]) for _ in range(len(path)-1)}
                    # 注意，这里计算conflict_nodes 需要使用的半径是4，而不是2!!
                    conflict_nodes = {v_ for v in nodes for v_ in conflict_node(v,4)} 
                    # remove nodes from Vj
                    V_j = V_j - conflict_nodes
                    conflict_arcs  = {arc for arc in E_j if (arc[1] in conflict_nodes or arc[0] in conflict_nodes)}
                    # remove arcs from Ej
                    E_j = E_j - conflict_arcs
            # 使用Bellman-Ford算法计算新的路径
            dist,new_path=BF(V_j,E_j,pe,('S',0),('T',0)) 
            # 更新fea_sol
            fea_sol[train]=new_path
    return fea_sol


'''
x is a dictionary, x[j] is a dictionary, x[j][arc] is the value of x_j(arc)
Ej is a dictionary, Ej[j] is the set of arcs of train j
V is the set of nodes
N is the dictionary, N[v] is the set of neighbors of node v
ignore is a set (default empty), ignore the train in this set.
'''
def compute_y_and_Y(x, Ej, num_train, V, N,ignore=set()):
    # Compute y_v
    y = {v: 0 for v in V}
    for _ in range(1, num_train + 1):
        if _ in ignore:
            continue
        for arc in Ej[_]:
            if x[_][arc] == 0: continue
            if (to := arc[1]) != ('T', 0): y[to] += 1

    # Compute Y_v
    Y = {v: sum(y[v_] for v_ in N[str(v)]) for v in V}

    return y, Y


'''
customized augumented Lagrangian method
'''

def ALM(param,num_train=16,k_max=10,t_max=1,mu=10,alpha=10):
    # 读取参数
    pe=param['pe']
    N=param['N']
    V=param['nodes']
    V_ = set(V)
    V_.update({('S',0),('T',0)}) # add artificial nodes
    E=param['arcs']
    Ej={j:param[idx(j)]['arcs'] for j in range(1,num_train+1)}

    # 初始化乘子lmd和rho
    lmd={v:0 for v in V}
    rho=2
    # 初始化目标函数值为-inf
    objval=float('-inf')
    # 初始化最初的x
    x={j:{arc:0 for arc in Ej[j]} for j in range(1,num_train+1)} # 全0初始化
    x_={}
    # 开始外层循环
    for k in tqdm(range(k_max)):
        # 开始内层循环
        for t in range(t_max):
            for j in range(1,num_train+1): # j is the NO of train
                # 为了更新x_j, 我们必须算出此时增广拉格朗日函数对x_j的梯度
                # 先计算y_v,Y_v , 忽略第j个train
                y,Y=compute_y_and_Y(x,Ej,num_train,V,N,ignore={j})
                # 下面计算BCD迭代需要的权重, 首先考虑来自原始目标函数的部分
                weights={arc:pe[arc] for arc in Ej[j]}
                # 下面考虑来自对偶目标函数的部分
                for arc in Ej[j]:
                    if (to := arc[1]) != ('T', 0):  # 如果是指向虚拟终点的边，这些边不会出现在对偶目标函数中
                        #grad[arc] -= sum(lmd[v] + rho * max(0, Y[v]-1) for v in N[str(to)]) proximal linear 的方法，效果不好
                        weights[arc] -= sum(lmd[v]+rho*max(0,Y[v]-1/2)  for v in N[str(to)])
                

                new_path=BF(V_,Ej[j],weights,('S',0),('T',0))[1]
                if len(new_path)==0 : print('error occurs at BF in BCD') # just for debug
                # 开始更新x_j
                x[j]={arc:0 for arc in Ej[j]}
                x[j][(('S',0),(new_path[0]))]=1
                for _ in range(1,len(new_path)):
                    x[j][(new_path[_-1],new_path[_])]=1
        
        #检查BCD得到的解是否可行   
        y,Y=compute_y_and_Y(x,Ej,num_train,V,N)
        if max(Y.values())>1 : # BCD的解不可行
            print('BCD not feasible, use heuristic!')
            new_sol=find_feasible(param,lmd,vec_to_path(x),num_train)
            x_= path_to_vec(new_sol,param)  
        else: 
            x_=copy.deepcopy(x) 
        # 更新目标函数值 
        new_objval=0
        for train in range(1,num_train+1):
            new_objval+=sum(pe[arc]*x_[train][arc] for arc in Ej[train])
        if new_objval>objval:
            objval=new_objval
            x_star=x_
        # update multipliers and penalty coefficients
        print('objval:',objval)
        for _ in lmd.keys():
            lmd[_]=max(0,lmd[_]+alpha*(Y[_]-1))
        rho+=alpha/2*sum(max(0,value-1)**2 for value in Y.values())
    return x_star,objval



                





            
    
    
    







