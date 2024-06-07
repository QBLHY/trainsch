import json
import matplotlib.pyplot as plt

#  read solution from solution.json
with open('./data/gb_solution.json', 'r') as f:
    sol_arcs = json.load(f)

times={} # time at each station of each train
for i in range(1,17): # 16 trains
    times[i]=[sol_arcs[str(i)][j][1] for j in range(1,len(sol_arcs[str(i)]))]
    

# 创建一个新的图形
fig, ax = plt.subplots()

# 设置横坐标的范围和间隔
ax.set_xticks(range(0, 161, 10))

# 设置纵坐标的标签
y_ticks = range(7)
ax.set_yticks(y_ticks)
ax.set_yticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G'])

# 设置横坐标和纵坐标的范围
ax.set_xlim([0, 160])
ax.set_ylim([-0.2, 6])

# 在每个纵坐标位置画水平线
for y in y_ticks:
    ax.axhline(y, color='lightgray', linewidth=0.5)

# 在横坐标位置画垂直线
for x in range(0, 170, 10):
    ax.axvline(x, color='lightgray', linewidth=0.5)

# 画出每列车的运行时间
station_index=[0,1,1,2,2,3,3,4,4,5,5,6]
for i in range(1, 17):
    ax.plot(times[i], station_index)
    ax.text(times[i][0]-0.5, -0.2, f'{2*i-1}', va='bottom',fontsize=6)


# 设置横纵坐标的标签
plt.xlabel('Time')
plt.ylabel('Station')

# 设置标题
plt.title('Train Schedule')

# 保存图像
plt.savefig('./pic/gurobi.png')

