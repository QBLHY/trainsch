# Project  on  Railway  Timetabling
This implementation is by Yirong Hu and Haotian He.

## Data Preparation
The `model_param` function within the `prepare.py` script facilitates the creation of the space-time network model along with the associated data.

## Problem 1 : Solve by Gurobi

### Setting 1 : arrange all trains on the schedule

```
python gurobi.py --setting 1
```

Now the solution is in the file `/data/gb_solution.json`.

Then you can use:
```
python visual_tool/visual1.py --solution_name gb_solution_1 --pic_name gb1
```
to plot the timetable in the file `/pic/gb1.png`

### LP relaxation of Setting 1
```
python gurobiLP.py --setting 1
```

The solution of LP relaxation is not binary, thus is not a valid timetable, so we don't save the solution of LP relaxation.

### Setting 2 : arrange all trains on the schedule  with the requirement of minimizing the total stoppage time. 

The usage process is identical to that of Setting 1, requiring only to replace "--setting 1" with "--setting 2" or to just omit it (as Setting 2 is the default). 

### LP relaxation of Setting 2
```
python gurobiLP.py --setting 2
```
## Problem 2

## Problem 3


## Problem 4 : Augmented Lagrangian Method (ALM) For Setting 1,2

For Setting 1
```
python ALM.py --setting 1 --max_iter 5

```
For Setting 2
```
python ALM.py --setting 2 --max_iter 60
```
This command will start the iteration of ALM algorithm. In setting 1, the optimal solution is found after the first iteration. In setting 2 , a near-optimal solution is found after 50 iterations. **Please wait for the program to finish executing; otherwise, the results will not be saved.**

After the execution, the solution is saved in the folder `/data`. Then you can run the following command to visualize the timetable.

```
python visual_tool/visual1.py --solution_name alm_solution_1 --pic_name alm1
```

The picture of timetable is in `/pic/alm1.png`.  

Setting 2 is similar.