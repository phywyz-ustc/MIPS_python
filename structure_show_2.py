import numpy as np
import matplotlib.pyplot as plt

rous = np.load('D:/pythonhomework/rous_mips.npy')/0.02
ACTS = ['0', '0.05', '0.1', '0.15', '0.2']
DRS  = ['1', '2', '5']
x = np.zeros(75)
for i in range(75):
    x[i] = i*0.02
#(5,3,75)
plt.figure(figsize=(20, 10))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.plot(x,rous[i, 0, :], label=f'Dr = {DRS[0]}')
    plt.plot(x,rous[i, 1, :], label=f'Dr = {DRS[1]}')
    plt.plot(x,rous[i, 2, :], label=f'Dr = {DRS[2]}')
    plt.title(f'Rous: v = {ACTS[i]}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.plot(x,rous[0, i, :], label=f'v = {ACTS[0]}')
    plt.plot(x,rous[1, i, :], label=f'v = {ACTS[1]}')
    plt.plot(x,rous[2, i, :], label=f'v = {ACTS[2]}')
    plt.plot(x,rous[3, i, :], label=f'v = {ACTS[3]}')
    plt.plot(x,rous[4, i, :], label=f'v = {ACTS[4]}')
    
    plt.title(f'Rous: Dr = {DRS[i]}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
plt.tight_layout()
plt.show()