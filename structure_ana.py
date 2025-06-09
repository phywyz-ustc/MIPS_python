import numpy as np
from numba import jit
import matplotlib.pyplot as plt
num_k   = 30  #int
num_particles = 2048 #int
box_length = 57 #float64
dk = 2*np.pi/box_length

@jit(nopython=True)
def compute_sk(positions, num_k, dk_x, dk_y , num_particles, box_length):
    sk = np.zeros(num_k)
    for j in range(num_k):
        k_x = (j+1)*dk_x
        k_y = (j+1)*dk_y
        for m in range(num_particles):
            sk_real = 0
            sk_fake = 0
            for n in range(num_particles):
                dx = positions[n,0]- positions[m,0]
                dx -= np.round(dx/box_length)*box_length
                dy = positions[n, 1] - positions[m, 1]
                dy -= np.round(dy / box_length) * box_length          

                delt_real = np.cos(k_x*dx + k_y*dy) 
                delt_fake = np.sin(k_x*dx + k_y*dy)
                sk_real += delt_real
                sk_fake += delt_fake
            sk[j] += (sk_real**2+ sk_fake**2)/num_particles
    return sk/num_particles

vlistn = np.array([0, 0.05, 0.1, 0.15, 0.2])
vlist = ['0.0', '0.05', '0.1', '0.15', '0.2']
vindex = 0

dlistn = np.array([ 1, 2, 5])
dlist = [ '1.0', '2.0', '5.0']
dindex = 0

seedlist = ['101', '102', '103', '104', '105', '106', '107', '108', '109', '110']

sks = np.zeros((len(vlist),len(dlist), num_k))


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
            sk1 = compute_sk(positions,num_k, dk, 0, num_particles, box_length)
            sk2 = compute_sk(positions,num_k, 0, dk, num_particles, box_length)
            sk3 = compute_sk(positions,num_k, dk/np.sqrt(2),dk/np.sqrt(2), num_particles, box_length)
            sks[vindex, dindex] += (sk1+sk2+sk3)/3
        sks[ vindex, dindex] /= len(seedlist)
        dindex += 1
    vindex += 1

ks = np.zeros(num_k)
for i in range(num_k):
    ks[i] = (i+1)*dk

np.save('sks_mips.npy',sks)
np.save('ks.npy',ks)