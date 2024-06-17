from prepare import model_param
from algorithms import ALM,vec_to_path,path_to_vec
import json
import argparse

def main(setting):
    param=model_param(setting=setting)
    x,val=ALM(param,num_train=16,k_max=70,t_max=1,alpha=1)
    path=vec_to_path(x)
    for i in range(1,17):
        path[i].insert(0,('S',0))
        path[i].pop() # remove the last station

    # save path
    with open(f'data/alm_solution_{setting}.json', 'w') as f:
        json.dump(path, f)
    print('the final obj val is', val)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=int, default=2)
    args = parser.parse_args()
    main(args.setting)