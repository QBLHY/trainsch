from prepare import model_param
from algorithms import ALM,vec_to_path,path_to_vec
import json

param=model_param()
x,val=ALM(param,num_train=16,k_max=1,t_max=10,mu=0.1,alpha=1)
path=vec_to_path(x)

for i in range(1,17):
    path[i].insert(0,('S',0))
    path[i].pop() # remove the last station

# save path
with open('data/alm_solution.json', 'w') as f:
    json.dump(path, f)
