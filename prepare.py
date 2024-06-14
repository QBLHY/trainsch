'''
Create Data for the Space-Time network model
'''


import json


'''
Model Parameters
'''
# running time between stations
RT=[(10,9),(20,18),(14,12),(8,7),(8,7),(10,8)]
RANGE=[(0,99),(9,108),(27,126),(39,138),(46,145),(53,152),(61,160)]

FAST=6
SLOW=10
# status 记录速度车在BCDEF站是否需要停，第一个列表代表速度为300的10辆车，后面6个列表代表快车
STATUS=[[1,1,1,1,1],[0,0,0,0,1],[1,1,0,1,0],[1,0,1,0,1],
        [1,0,0,1,0],[1,0,0,0,1],[1,0,0,1,0]]


'''
Calculate all possible arcs , i.e. E
'''
def all_arcs():
    arcs=[]
    # from S to A
    for i in range(RANGE[0][0],RANGE[0][1]+1):
        arcs.append((('S',0),('A',i)))
    # from G to T 
    for i in range(RANGE[6][0],RANGE[6][1]+1):
        arcs.append((('G',i),('T',0)))
    all_pos = ['A','B1','B2','C1','C2','D1','D2','E1','E2','F1','F2','G'] # len = 12
    for i in range(11):
        pos=all_pos[i]
        idx=(i+1)//2 # A is 0, B is 1 ,...
        next_pos=all_pos[i+1]
        if i%2==0: # cross station
            for t in range(RANGE[idx][0],RANGE[idx][1]+1):
                for delta_t in [RT[idx][0],RT[idx][1]]:
                    if t+delta_t in range(RANGE[idx+1][0],RANGE[idx+1][1]+1):
                        arcs.append(((pos,t),(next_pos,t+delta_t)))
        else : # same station
            # arcs represent not stopping in this station
            for t in range(RANGE[idx][0],RANGE[idx][1]+1):
                for delay in range(2,16):
                    if t+delay in range(RANGE[idx][0],RANGE[idx][1]+1):
                        arcs.append(((pos,t),(next_pos,t+delay)))
            # arcs represent  stopping in this station
            for t in range(RANGE[idx][0],RANGE[idx][1]+1):
                arcs.append(((pos,t),(next_pos,t)))
    return arcs


'''
Calculate available arcs for a train, i.e. E^j
'''
def available_arcs(idx): # has arcs related to artificial nodes
    # idx 是车辆的编号，1，2，3，4，5，6代表6个快车。其他是慢车
    arcs=[]
    spd=int(1<=idx and idx<=6) # 快车是1，慢车是0
    r=RANGE
    for i in range(8):
        if i>=2 and i<=6:
            this=chr(64+i)
            next=chr(65+i)
            # 是否在this 停靠
            if STATUS[idx][i-2]==1: # 需要停靠
                for time in range(r[i-1][0],r[i-1][1]+1):
                    for delay in range(2,16):
                        if time+delay<=r[i-1][1]:
                            arcs.append(((this+'1',time),(this+'2',time+delay)))
            else : # 不需要停靠
                for time in range(r[i-1][0],r[i-1][1]+1):
                    arcs.append(((this+'1',time),(this+'2',time)))
            # 从 this 到 next 
            for time in range(r[i-1][0],r[i-1][1]+1):
                if i==6 : # F to G 
                    if (next_time:=time+RT[i-1][spd]) in range(r[i][0],r[i][1]+1):
                        arcs.append(((this+'2',time),(next,next_time)))
                else:
                    if (next_time:=time+RT[i-1][spd]) in range(r[i][0],r[i][1]+1):
                        arcs.append(((this+'2',time),(next+'1',next_time))) 
        elif i==0: # start from S
            for j in range(r[0][0],r[0][1]+1):
                arcs.append((('S',0),('A',j)))
        elif i==1: # from A to B 
            for j in range(r[0][0],r[0][1]+1):
                if j+RT[0][spd] in range(r[1][0],r[1][1]+1):
                    arcs.append((('A',j),('B1',j+RT[0][spd])))
        elif i==7: # from G to end
            for j in range(r[6][0],r[6][1]+1):
                arcs.append((('G',j),('T',0)))
    return arcs

def available_nodes(): # NO ARTIFICIAL NODES
    r=RANGE
    list=[]
    for i in range(7):
        for j in range(r[i][0],r[i][1]+1):
            if i==6:
                list.append(('G',j))
            elif i==0:
                list.append(('A',j))
            else:
                list.append((chr(i+65)+'1',j))
                list.append((chr(i+65)+'2',j))
    return list


'''
Caculate delta^+_j(v) and delta^-_j(v)
'''
def delta(v,idx,out=True): # support artificial nodes
    # out=True, delta^+_j(v), out=False, delta^-_j(v)
    # v is node, v is a tuple of (position,time) 
    # idx is the number of train, 1,2,3,4,5,6 fast ,0 for slow
    list=[]
    spd=int(1<=idx and idx<=6) # 快车是1，慢车是0
    r=RANGE
    # parse node v
    pos,time=v
    if out:
        if pos=='T': # at end station
            return []
        if pos=='G':
            return [(('G',time),('T',0))]
        if pos=='S': # at start station
            for t in range(r[0][0],r[0][1]+1):
                list.append((('S',0),('A',t)))
            return list
        if pos=='A':
            if time + RT[0][spd] in range(r[1][0],r[1][1]+1):
                return [(('A',time),('B1',time+RT[0][spd]))]
            else :return []
        # in the other cases, pos is a string of len=2
        id=ord(pos[0])-65 # index of this station ,'B' is 1
        if pos[1]=='2': # travel to next station
            next_pos=chr(ord(pos[0])+1)+'1' if pos[0]!='F' else 'G'
            next_time=time+RT[id][spd]
            if next_time in range(r[id+1][0],r[id+1][1]+1):
                return [((pos,time),(next_pos,next_time))]
            else : return []
        else : # in the same station
            # check whether need to stop at this station
            next_pos=pos[0]+'2'
            if STATUS[idx][id-1]==1: # need to stop
                for delay in range(2,16):
                    next_time=time+delay
                    if next_time in range(r[id][0],r[id][1]+1):
                        list.append(((pos,time),(next_pos,next_time)))
                return list
            else: # don't stop
                return [((pos,time),(next_pos,time))]
    else:  # out = False
        if pos=='S':
            return []
        if pos=='A':
            return [(('S',0),('A',time))]
        if pos=='T':
            for t in range(r[6][0],r[6][1]+1):
                list.append((('G',t),('T',0)))
            return list
        if pos=='G':
            last_time=time-RT[5][spd]
            if last_time in range(r[5][0],r[5][1]+1):
                return [(('F2',last_time),('G',time))]
            else : return []
        # in the other cases, pos is a string of len=2
        id=ord(pos[0])-65 # index of this station, 'B' is 1
        if pos[1]=='1': # come from the last station
            last_pos=chr(ord(pos[0])-1)+'2' if pos[0]!='B' else 'A'
            last_time=time-RT[id-1][spd]
            if last_time in range(r[id-1][0],r[id-1][1]+1):
                return[((last_pos,last_time),(pos,time))]
            else : return []
        else : # in the same station
            last_pos=pos[0]+'1'
            # check whether need stop
            if STATUS[idx][id-1]==1: # need to stop
                for delay in range(2,16):
                    last_time=time-delay
                    if last_time in range(r[id][0],r[id][1]+1):
                        list.append(((last_pos,last_time),(pos,time)))
                return list
            else: # don't stop
                return[((last_pos,time),(pos,time))]


def conflict_node(v):
    pos, time = v
    if pos in {'S', 'T'}:
        return []
    index = ord(pos[0]) - 65 # index of this station, 'B' is 1
    conflict = [(pos, t+time) for t in range(-2, 3) if t+time in range(RANGE[index][0], RANGE[index][1]+1)]
    return conflict

def init_train(idx):
    # train is a dictionary of dictionary
    nodes=available_nodes()
    arcs=available_arcs(idx)
    delta_plus={}
    delta_minus={}
    for v in nodes:
        delta_plus[str(v)]=delta(v,idx,out=True)
        delta_minus[str(v)]=delta(v,idx,out=False)
    delta_plus[str(('S',0))]=delta(('S',0),idx,out=True)
    delta_plus[str(('T',0))]=[]
    delta_minus[str(('T',0))]=delta(('T',0),idx,out=False)
    delta_minus[str(('S',0))]=[]
    train={
        "nodes":nodes,
        "arcs":arcs,
        "delta_plus":delta_plus,
        "delta_minus":delta_minus
    }
    return train

def model_param(): 
    train1=init_train(1)
    train2=init_train(2)
    train3=init_train(3)
    train4=init_train(4)
    train5=init_train(5)
    train6=init_train(6)
    train0=init_train(0)
    nodes=available_nodes() # has no artificial nodes
    Nv={}
    for node in nodes:
        Nv[str(node)]=conflict_node(node)
    modelparam={
        'N':Nv,
        'nodes':nodes, 'arcs':all_arcs(),
        "0":train0,"1":train1, "2":train2,"3":train3,
        "4":train4,"5":train5, "6":train6
    }
    pe={str(arc):0 for arc in modelparam['arcs']}
    for t in range(100):
        pe[str((('S',0),('A',t)))]=1 
    modelparam['pe']=pe
    return modelparam

if __name__ == '__main__':
    #Use json.dump to write data to a JSON file
    param=model_param()
    with open('data/modelparam.json', 'w') as f:
        json.dump(param, f)    



        
        
        
        

    
