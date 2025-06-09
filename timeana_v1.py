import numpy as np
import re
import matplotlib.pyplot as plt

DR_VALUES = [ 1, 2, 5]
BOX_VALUES = [20, 28, 40, 57, 80, 113, 160]
SEED_VALUES = list(range(101, 111))
Num_Values = [256, 512, 1024, 2048, 4096, 8192, 16384]
Nlogn = Num_Values*np.log(Num_Values)
cost_array = np.full((len(DR_VALUES), len(BOX_VALUES), len(SEED_VALUES)), np.nan)

dr_idx = {v: i for i, v in enumerate(DR_VALUES)}
box_idx = {v: i for i, v in enumerate(BOX_VALUES)}
seed_idx = {v: i for i, v in enumerate(SEED_VALUES)}

with open('D:/pythonhomework/timesrecord_withoutoutput_v1.txt', 'r') as f:
    for line in f:
        match = re.search(
            r"seed=(\d+\.\d+),v=(\d+\.\d+),box=(\d+\.\d+),Dr=(\d+\.\d+)\s+cost:\s+([\d\.]+)",
            line
        )
        if match:
            seed = int(float(match.group(1)))
            box = int(float(match.group(3)))
            dr = float(match.group(4))
            cost = float(match.group(5))

            if dr in dr_idx and box in box_idx and seed in seed_idx:
                i = dr_idx[dr]
                j = box_idx[box]
                k = seed_idx[seed]
                cost_array[i, j, k] = cost

mean_cost = np.nanmean(cost_array, axis=2)

plt.figure(figsize=(10, 6))
for i, dr in enumerate(DR_VALUES):
    plt.plot(Num_Values, mean_cost[i], marker='o', label=f'Dr={dr}')

plt.xlabel("Number of Particles (N)")
plt.ylabel("Mean Cost over Seeds")
plt.title("Cost vs N for Different Dr (v=0.1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for i, dr in enumerate(DR_VALUES):
    plt.plot(Nlogn, mean_cost[i], marker='o', label=f'Dr={dr}')

plt.xlabel("N log(N)")
plt.ylabel("Mean Cost over Seeds")
plt.title("Cost vs NlogN for Different Dr (v=0.1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
