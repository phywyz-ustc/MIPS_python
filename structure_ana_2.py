import numpy as np
from numba import jit
import matplotlib.pyplot as plt

num_particles = 2048 #int
box_length = 57 #float64
R = 0.15*box_length #float64

@jit(nopython=True)
def compute_rou(positions, num_particles, box_length, R):
    rou = np.zeros(75)
    rous_by_particles = np.zeros(num_particles)
    for i in range(num_particles):
        for j in range(num_particles):
            if i >= j:
                continue
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dx -= np.round(dx / box_length) *box_length
            dy -= np.round(dy / box_length) *box_length
            dis = np.sqrt(dx**2 + dy**2 )
            if dis < R:
                rous_by_particles[i] += 1
                rous_by_particles[j] += 1
    rous_by_particles /= (np.pi * R**2)
    for i in range(num_particles):
        rouindex = int(rous_by_particles[i]/0.02)
        if rouindex >= 75:
            print(f'density of {i} is larger than or equals 1.5')
        rou[rouindex] += 1
    return rou/num_particles

vlistn = np.array([0, 0.05, 0.1, 0.15, 0.2])
vlist = ['0.0', '0.05', '0.1', '0.15', '0.2']
vindex = 0

dlistn = np.array([ 1, 2, 5])
dlist = [ '1.0', '2.0', '5.0']
dindex = 0

seedlist = ['101', '102', '103', '104', '105', '106', '107', '108', '109', '110']

rous = np.zeros((len(vlist),len(dlist), 75))


flag = 0

for act in vlist:
    dindex = 0
    for dr in dlist:
        print(f'v={act} dr={dr}')
        for seed in seedlist:
            path = f"/data/yzwang/cell_md/mipsrecord/records_v_{act}_Dr_{dr}_Dt_0.1_seed_{seed}_nump_2048_box_57"
            path2 = '.npy'
            pars_record = np.load(path+path2)
            if flag==0:
                print(np.shape(pars_record))
                flag =1

            positions = pars_record[-1]
            rou = compute_rou(positions, num_particles, box_length, R)
            rous[vindex, dindex] += rou

        rous[ vindex, dindex] /= len(seedlist)
        dindex += 1
    vindex += 1

np.save('rous_mips.npy',rous)