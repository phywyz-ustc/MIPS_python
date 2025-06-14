import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt

@njit
def compute_neibs_cell(positions,num_particles,box,   bin_size, num_bins):
    neighbors = np.zeros((num_particles, 16), dtype=np.int32)  # 假设每个粒子最多有 100 个邻居
    neighbors_len = np.zeros(num_particles, dtype=np.int32)  # 每个粒子的邻居数

    binlist = np.full((num_bins * num_bins, 4), -1, dtype=np.int32)
    bincount = np.zeros(num_bins * num_bins, dtype=np.int32)

    inbin_x = np.zeros(num_particles, dtype=np.int32)
    inbin_y = np.zeros(num_particles, dtype=np.int32)

    for i in range(num_particles):

        x_idx = int(np.floor(positions[i, 0] // bin_size)%num_bins)
        y_idx = int(np.floor(positions[i, 1] // bin_size)%num_bins)
        bin_index = x_idx * num_bins + y_idx
        inbin_x[i] = x_idx
        inbin_y[i] = y_idx
        pos_in_bin = bincount[bin_index]
        binlist[bin_index, pos_in_bin] = i
        bincount[bin_index] += 1

    for i in range(num_particles):
        x_idx = inbin_x[i]
        y_idx = inbin_y[i]
        #print('i:', i)
        #print('xy:', x_idx, y_idx)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx = (x_idx + dx + num_bins) % num_bins
                ny = (y_idx + dy + num_bins) % num_bins
                #print('nxy:', nx, ny)
                neighbor_bin = nx * num_bins + ny

                for j in range(bincount[neighbor_bin]):
                    nbr = binlist[neighbor_bin, j]
                    if nbr == i:
                        continue
                    n_now = neighbors_len[i]
                    neighbors[i, n_now] = nbr
                    neighbors_len[i] += 1
    return neighbors, neighbors_len

def initial(seed, num_particles , num_bins):
    np.random.seed(seed)
    grid_indices = np.random.choice(num_bins * num_bins, num_particles, replace=False)
    positions = np.stack( (grid_indices//num_bins, grid_indices% num_bins), axis = 1)+0.5

    return positions

if __name__ == "__main__":
    num_particles = 1024
    num_bins = 40
    seed = 101
    particle_coords = initial(seed, num_particles , num_bins)
    print("粒子坐标示例：")
    print(particle_coords[:10])
    plt.figure(figsize=(8, 8))
    plt.scatter(particle_coords[:, 0], particle_coords[:, 1], color='blue', s=10)
    plt.title("40x40 Grid Particle Distribution")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(-1, 30)
    plt.ylim(-1, 30)
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.show()
