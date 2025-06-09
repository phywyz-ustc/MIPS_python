import numpy as np
import matplotlib.pyplot as plt

sks = np.load('D:/pythonhomework/sks_mips.npy')
ks  = np.load('D:/pythonhomework/ks.npy')
ACTS = ['0', '0.05', '0.1', '0.15', '0.2']
DRS  = ['1', '2', '5']
sk_log_s = np.log(sks)
ks_log = np.log(ks)

ys = np.max(sk_log_s)+0.5
xs = ks_log[0]
xs2= xs+0.2
ys2= ys-0.4


plt.figure(figsize=(20, 10))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.plot([xs, xs2], [ys, ys2], 'k--', lw=2)
    plt.text(xs2, ys2, 'k^(-2)', fontsize=20, color='black')
    plt.plot(ks_log, sk_log_s[i, 0, :], label=f'Dr = {DRS[0]}')
    plt.plot(ks_log, sk_log_s[i, 1, :], label=f'Dr = {DRS[1]}')
    plt.plot(ks_log, sk_log_s[i, 2, :], label=f'Dr = {DRS[2]}')
    plt.title(f'log(sk) - log(k): v = {ACTS[i]}')
    plt.xlabel('log(k)')
    plt.ylabel('log(sk)')
    plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.plot([xs, xs2], [ys, ys2], 'k--', lw=2)
    plt.text(xs2, ys2, 'k^(-2)', fontsize=20, color='black')
    plt.plot(ks_log,sk_log_s[0, i, :], label=f'v = {ACTS[0]}')
    plt.plot(ks_log,sk_log_s[1, i, :], label=f'v = {ACTS[1]}')
    plt.plot(ks_log,sk_log_s[2, i, :], label=f'v = {ACTS[2]}')
    plt.plot(ks_log,sk_log_s[3, i, :], label=f'v = {ACTS[3]}')
    plt.plot(ks_log,sk_log_s[4, i, :], label=f'v = {ACTS[4]}')
    
    plt.title(f'log(sk) - log(k): Dr = {DRS[i]}')
    plt.xlabel('log(k)')
    plt.ylabel('log(sk)')
    plt.legend()
plt.tight_layout()
plt.show()