import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from moviepy import ImageSequenceClip
from compute import compute_neibs_cell, initial
import os
import sys

@njit
def update(positions, thetas, Dr, Dt,  box, dt, bin_size, num_bins, num_particles, cut, sigma, eps, v):
    drfs = np.random.normal(0, 1, num_particles) * np.sqrt(2 * Dr)
    dtfs = np.random.normal(0, 1, (num_particles, 2)) * np.sqrt(2 * Dt)
    disp = np.zeros((num_particles, 2), dtype=np.float64)
    neighbors, neighbors_len = compute_neibs_cell(positions , num_particles, box, bin_size, num_bins)
    for i in range(num_particles):
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
    positions += disp
    positions += dtfs*dt
    thetas += drfs* dt
    thetas -= np.floor(thetas/(2*np.pi))*2*np.pi
    acts = np.stack((np.cos(thetas), np.sin(thetas)), axis=1)
    positions += v * acts * dt
    positions -= np.floor(positions/box)*box

def save_frame(step, positions, thetas, box, fig, ax, frames_dir, areas, dpi):
    ax.clear()
    ax.set_xlim(0, box)
    ax.set_ylim(0, box)
    ax.set_aspect('equal')
    ax.axis('off')

    colors = np.cos(thetas)

    ax.scatter(
        positions[:, 0], positions[:, 1],
        s=areas,
        c=colors,
        cmap='coolwarm',
        vmin=-1, vmax=1,
        edgecolors='none'
    )
    ax.set_title(f"Step {step}")

    os.makedirs(frames_dir, exist_ok=True)
    frame_path = os.path.join(frames_dir, f"frame_{step:05d}.png")
    fig.savefig(frame_path, dpi=dpi)
    return frame_path

def main():
    dt = 0.01
    num_steps = 64*2048+1
    cut = 1  
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
    bin_size = 1
    num_bins = int(box) 
    # 初始化
    np.random.seed(seed)
    positions = initial(seed, num_particles, num_bins)  
    thetas = np.random.uniform(0, 2 * np.pi, size=num_particles)
    frames = []

    fig, ax = plt.subplots(figsize=(6, 6),  dpi=300)

    ax_size_inches = fig.get_size_inches()
    dpi = fig.dpi
    width_px = ax_size_inches[0] * dpi
    pixels_per_unit = width_px / box
    radius = 0.5 * pixels_per_unit * 72 / dpi
    areas = np.pi * radius ** 2
    
    for step in range(num_steps):
        update(positions, thetas, Dr, Dt, box, dt, bin_size, num_bins, num_particles, cut, sigma, eps, v)
        #print('done')
        if step % 50 == 0:
            print(f"Saving frame at step {step}")
            frame_path = save_frame(step, positions, thetas, box, fig, ax, "mipsvediosmake/frames"+"_v_"+str(v)+"_Dr_"+str(Dr)+"_Dt_"+str(Dt)+"_seed_"+str(seed)+"_nump_"+str(num_particles)+"_box_"+str(box), areas, dpi)
            frames.append(frame_path)

    plt.close(fig)
    output_path = "/data/yzwang/cell_md/mipsvedios/output"+"_v_"+str(v)+"_Dr_"+str(Dr)+"_Dt_"+str(Dt)+"_seed_"+str(seed)+"_nump_"+str(num_particles)+"_box_"+str(box)+".mp4"
    print("Compiling video...")
    clip = ImageSequenceClip(frames, fps=24)
    clip.write_videofile(output_path,  audio=False)
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    main()
