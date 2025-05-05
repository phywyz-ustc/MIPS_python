import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from moviepy import ImageSequenceClip
from compute import compute_neibs_cell, initial
import os
import sys

@njit
def update(positions, thetas, drfs, dtfs,  box, dt, bin_size, num_bins, num_particles, cut, sigma, eps, v):
    disp = np.zeros((num_particles, 2), dtype=np.float64)
    #print('before')
    neighbors, neighbors_len = compute_neibs_cell(positions , num_particles, box, bin_size, num_bins)
    #print('after')
    for i in range(num_particles):
        #print('i:',i)
        #print(neighbors[i])
        for k in range(neighbors_len[i]):
            nbr = neighbors[i][k]
            if i >= nbr:
                continue
            dxy = positions[nbr] - positions[i]
            dxy -= np.round(dxy / box) * box
            dd = np.sqrt(np.sum(dxy**2))
            if dd < cut:
                ratio = (sigma/ dd)**6
                force   =  -24 * eps *(2*ratio**2 - ratio   ) * dxy/dd**2
                disp[i] += force * dt
                disp[nbr] -= force * dt
    #print('over')
    # 积分更新
    positions += disp
    positions += dtfs*dt
    thetas += drfs* dt
    thetas -= np.floor(thetas/(2*np.pi))*2*np.pi
    acts = np.stack((np.cos(thetas), np.sin(thetas)), axis=1)
    positions += v * acts * dt
    positions -= np.floor(positions/box)*box

def main():
    dt = 0.01
    block_size = 11
    block_size_real = np.power(2,block_size-1)
    num_blocks = 100
    num_steps  = num_blocks * block_size_real+1
    print(num_steps)
    recordtimes = np.zeros(num_blocks*block_size)
    for i in range(num_blocks):
        for j in range(block_size):
            idx = i*block_size+j
            recordtimes[idx] = i*block_size_real +np.power(2, j)
            print('idx:', recordtimes[idx])

    cut = 1  # WCA 势的截断半径
    sigma = 1 / 2 ** (1 / 6)
    eps = 0.1
    
    # 读取命令行参数
    v = float(sys.argv[1])
    Dr = float(sys.argv[2])
    Dt = float(sys.argv[3])
    seed = int(sys.argv[4])
    num_particles = int(sys.argv[5])
    box = int(sys.argv[6])
    
    print('v:', v)
    print('Dr:', Dr)
    print('Dt:', Dt)
    print('seed:', seed)
    print('num_particles:', num_particles)
    print('box:',box)
    records = np.zeros((num_blocks*block_size, num_particles,2))
    bin_size = 1
    num_bins = int(box)
    tt=0
    # 初始化
    np.random.seed(seed)
    positions = initial(seed, num_particles, num_bins)  # 从 compute.py 导入的初始函数
    thetas = np.random.uniform(0, 2 * np.pi, size=num_particles)
    for step in range(num_steps):
        # 随机扰动

        drfs = np.random.normal(0, 1, num_particles) * np.sqrt(2 * Dr)
        dtfs = np.random.normal(0, 1, (num_particles, 2)) * np.sqrt(2 * Dt)

        update(positions, thetas, drfs, dtfs, box, dt, bin_size, num_bins, num_particles, cut, sigma, eps, v)
        #print('done')
        if step  == recordtimes[tt]:
            records[tt] = positions
            print(step)
            tt+=1

    recordpath = "records"+"_v_"+str(v)+"_Dr_"+str(Dr)+"_Dt_"+str(Dt)+"_seed_"+str(seed)+"_nump_"+str(num_particles)+"_box_"+str(box)+".npy"
    np.save(recordpath, records)
if __name__ == "__main__":
    main()
