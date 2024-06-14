import copy
from tqdm import tqdm
import time 

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
def path_to_vec(path):
    x={}
    for train in path.keys():
        x[train]={}
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
    fea_sol={train:[] for train in range(1,num_train+1)}
    # compute dual obj function for each train
    dual_obj = {train:0 for j in range(1,num_train+1)}
    pe=param['pe']
    # train 为列车编号
    for train in range(1,num_train+1): 
        path = sol[str(train)]
        # 获得对应节点和边
        Ej = param[idx(train)]['arcs']
        for _ in range(len(path)-1):
            arc= tuple([tuple(path[_]),tuple(path[_+1])])
            t=arc[1] # arc=(*,t)
            # 计算这条边arc在dual obj function中前面对应的系数
            coeff=pe[str(arc)]
            if t[0]!='T':
                neighbors=param['N'][str(t)]
                for v in neighbors:
                    coeff-=mul[str(v)]
            dual_obj[train]+=coeff
    
    # 对dual_obj 进行排序
    sorted_dual = sorted(dual_obj.items(), key=lambda x: x[1],reverse=False)
    priority_list=[item[0] for item in sorted_dual]

    # 逐个替换列车
    done=set() # 已经安排好的列车集合
    for _ in range(5): # 最多执行5轮
        for train in priority_list:
            if train in done:
                continue
            # 获得对应节点和边
            Ej = copy.deepcopy(param[idx(train)]['arcs'])
            Vj = copy.deepcopy(param[idx(train)]['nodes'])
            # 去掉与其他列车冲突的node以及arc
            for t in range(1,num_train+1):
                if t!=train:
                    path = sol[str(t)] # the path of the train t
                    if (len(path)==0): continue
                    # nodes on this path
                    nodes={tuple(path[_]) for _ in range(len(path)-1)}
                    conflict_nodes = {v_ for v in nodes for v_ in param['N'][str(v)]}
                    conflict_arcs  = {arc for arc in Ej if arc[1] in conflict_nodes}
                    # remove nodes from Vj
                    Vj = set(Vj) - conflict_nodes
                    # remove arcs from Ej
                    Ej = set(Ej) - conflict_arcs
                    # 使用Bellman-Ford算法计算新的路径
            dist,new_path=BF(Vj,Ej,pe,tuple(path[0]),tuple(path[-1]))
            # 如果新的路径不为空，则更新fea_sol
            if new_path:
                fea_sol[train]=new_path
                done.add(train)
    if len(done)==16:   return fea_sol
    else:               return None
        
'''
x is a dictionary, x[j] is a dictionary, x[j][arc] is the value of x_j(arc)
Ej is a dictionary, Ej[j] is the set of arcs of train j
V is the set of nodes
N is the dictionary, N[v] is the set of neighbors of node v
'''
def compute_y_and_Y(x, Ej, num_train, V, N):
    # Compute y_v
    y = {v: 0 for v in V}
    for _ in range(1, num_train + 1):
        for arc in Ej[_]:
            if x[_][arc] == 0: continue
            if (to := arc[1]) != ('T', 0): y[to] += 1

    # Compute Y_v
    Y = {v: sum(y[v_] for v_ in N[str(v)]) for v in V}

    return y, Y


'''
customized augumented Lagrangian method
'''

def ALM(param,num_train=16,k_max=10,t_max=1,mu=0.1,alpha=1):
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
    
    BF_time =0 # TODO: remove this line
    update_time=0
    BCD_time=0
    Y_time=0
    grad_time=0
    grad1_time=0
    # 开始外层循环
    for k in tqdm(range(k_max)):
        # 开始内层循环
        for t in range(t_max):
            for j in range(1,num_train+1): # j is the NO of train
                # 为了更新x_j, 我们必须算出此时增广拉格朗日函数对x_j的梯度
                # 先计算y_v,Y_v
                y,Y=compute_y_and_Y(x,Ej,num_train,V,N)
                # 下面计算梯度, 首先考虑来自原始目标函数的部分
                grad={arc:pe[str(arc)] for arc in Ej[j]}
                # 下面考虑来自对偶目标函数的部分
                for arc in Ej[j]:
                    if (to := arc[1]) != ('T', 0):  # 如果是指向虚拟终点的边，这些边不会出现在对偶目标函数中
                        grad[arc] -= sum(lmd[v] + rho * max(0, Y[v]) for v in N[str(to)])
                # 下面计算需要传入给BF算法的weight
                weights={arc: mu*grad[arc]-1/2+x[j][arc] for arc in Ej[j]}

                new_path=BF(V_,Ej[j],weights,('S',0),('T',0))[1]
                
                
                if len(new_path)==0 : print('error occurs at BF in BCD') # just for debug
                # 开始更新x_j
                x[j]={arc:0 for arc in Ej[j]}
                x[j][(('S',0),(new_path[0]))]=1
                for _ in range(1,len(new_path)):
                    x[j][(new_path[_-1],new_path[_])]=1
        # 检查BCD得到的解是否可行
        
        if max(Y.values())>1: # BCD的解不可行
            new_sol=find_feasible(param,lmd,vec_to_path(x),num_train)
            if new_sol is None:
                print('error occurs at heuristic algorithm in ALM')
                return None
            x= path_to_vec(new_sol)  
        new_objval=0
        for train in range(1,num_train+1):
            new_objval+=sum(pe[str(arc)]*x[train][arc] for arc in Ej[train])
        if new_objval>objval:
            objval=new_objval
            x_star=x
        # update multipliers and penalty coefficients
        y,Y=compute_y_and_Y(x,Ej,num_train,V,N)
        for _ in lmd.keys():
            lmd[_]=max(0,lmd[_]+alpha*(Y[_]-1))
        rho+=alpha/2*sum(max(0,value-1)**2 for value in Y.values())
    
    return x_star,objval



                





            
    
    
    







