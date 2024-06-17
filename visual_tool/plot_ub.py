import matplotlib.pyplot as plt

# 读取文件内容
file_path = 'data\lagrangian_relax.txt'

epochs = []
target_values = []

with open(file_path, 'r') as file:
    for line in file:
        if line.startswith('epoch:'):
            parts = line.split()
            epoch = int(parts[1])
            target_value = float(parts[4])
            epochs.append(epoch)
            target_values.append(target_value)


plt.figure(figsize=(12, 6))
plt.plot(epochs, target_values, marker='o', linestyle='-', color='b')
plt.axhline(y=16, color='r', linestyle='--', label='y=16')

yticks = plt.yticks()[0]
new_yticks = list(yticks) + [16]
plt.yticks(new_yticks)
plt.title('Target Value vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Target Value')
plt.grid(True)
plt.xticks(range(0, 81, 5))
plt.savefig(f'./pic/plot_ub.png')