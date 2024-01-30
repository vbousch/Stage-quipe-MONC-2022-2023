import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy import interpolate

from model import model  # Assuming the model function is defined in a separate file

# Create a directory for results
folder = 'Results-test'
if not os.path.exists(folder):
    os.makedirs(folder)

# Description of N
Nend = 0.005  # 0.04
Nbegin = 0.04
dist = 0.3
tau = 0.05
p_of_N = [Nbegin, Nend, dist, tau]

# Time
tmax = 50 - 20

t_plot = np.array([0, 6, 12, 18, 24, 30, 36, 43, 66, 90, 114, 138, 162, 186, 199])  # linspace(0, tmax, 11)

# Initial condition
R0target = np.sqrt(0.4/np.pi)  # 0.084  # 0.35
P0target = 0.9

# Parameters
b = 0.095   # 0.01
k = 0.884   # 2.3

t, R, r01, P = model(tmax, R0target, P0target, b, k, p_of_N)
t += 20

with open('Rvstime.txt', 'w') as f:
    for j in np.arange(len(t)):
        f.write(str(t[j]) + ' ' + str(R[j]) + '\n')

for k in np.arange(len(t_plot)):
    with open('P_time' + str(t_plot[k]), 'w') as g:
        for i in np.arange(len(r01)):
            g.write(str(r01[i]) + ' ' + str(P[i, 134]) + '\n')

plt.figure()
plt.plot(t, R, color=[0, 0.4470, 0.7410], linewidth=3)
plt.xlabel('$t$ (h)', fontsize=14)
plt.ylabel('$R$ (mm)', fontsize=14)
plt.tick_params(labelsize=12)

plt.savefig(f'{folder}/R.pdf')

plt.figure()
plt.plot(t, np.pi*R*R, color=[0, 0.4470, 0.7410], linewidth=3)
plt.xlabel('$t$ (h)', fontsize=14)
plt.xlim(0, 72)
plt.ylabel('$Area$ (mm²)', fontsize=14)
plt.ylim(0.2, 1.)
plt.tick_params(labelsize=12)

plt.savefig(f'{folder}/Area.pdf')

# for i in range(len(t_plot)):
#     fig, axs = plt.subplots(1, 2, figsize=(16, 8))
#     p = np.argmin(np.abs(t - t_plot[i]))
#     axs[0].plot(r01, P[:, p], color=[0, 0.4470, 0.7410], linewidth=3)
#     axs[0].set_ylim([0, 1])
#     axs[0].set_xlabel('$r$', fontsize=18)
#     axs[0].set_title(f'$P$ at time {t_plot[i]:.1f} s', fontsize=20)
#     axs[0].tick_params(labelsize=20)
#     axs[1].plot(r01, 1 - P[:, p], color=[0, 0.4470, 0.7410], linewidth=3)
#     axs[1].set_ylim([0, 1])
#     axs[1].set_xlabel('$r$', fontsize=18)
#     axs[1].set_title(f'$Q$ at time {t_plot[i]:.1f} s', fontsize=20)
#     axs[1].tick_params(labelsize=20)
#     plt.tight_layout()
#     plt.savefig(f'{folder}/P_Q_{i+1}.pdf')


# def gompertz_model(t, a0, a, b):
#     A = a0 * np.exp(a/b * (1 - np.exp(-b*(t-20))))
#     return A
#
# dtypes = [('col0', 'U10'), ('col1', 'U10'), ('col2', float), ('col3', float),
#               ('col4', float), ('col5', float), ('col6', float), ('col7', float), ('col8', float)]
#
# with open('DataExtracted/results_ind.txt', 'r') as file:
#         lines = file.readlines()
#         params = {}
#         for line in lines[1:]:
#             cols = line.split()
#             id = int(cols[0])
#             A0 = float(cols[1])
#             a_area_gompertz = float(cols[2])
#             b_gompertz = float(cols[3])
#             params[id] = {'A0': A0, 'a_area_gompertz': a_area_gompertz, 'b_gompertz': b_gompertz}
#         file.close()
#
# data = np.loadtxt('DataExtracted/results_control_cleaned' + '.txt', dtype=dtypes, skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8))
# for Exp_ID in ["A2_", "A3_", "A4_", "A5_", "A6_", "A7_", "A8_", "A9_", "A10_"]:
#         Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
#         data_Exp = data[Exp_mask]
#         plt.plot(data_Exp['col2'], data_Exp['col7'], 'k.', markersize=1)
#         t_gomp = np.linspace(20., 50., 100)
#         a0 = params[int(Exp_ID[1:-1])]['A0']
#         a = params[int(Exp_ID[1:-1])]['a_area_gompertz']
#         b = params[int(Exp_ID[1:-1])]['b_gompertz']
#         gompertz = [gompertz_model(t, a0, a, b) for t in t_gomp]
#         mean_gompertz = [gompertz_model(t, 0.4, 0.056, 0.095) for t in t_gomp]
#         plt.plot(t_gomp, gompertz, 'black', lw=1, alpha=0.8)
# plt.plot(t_gomp, mean_gompertz, 'r', lw=2, label='fitting moyen')
# plt.plot(t, np.pi * R * R, color=[0, 0.4470, 0.7410], linewidth=2, label='simulation')
# plt.xlim(0., 72.)
# plt.ylim(0.2, 1.)
# plt.legend()
# plt.title('results_control')
# plt.savefig('volumes_plus_sim.pdf')






'''plots 2D'''
r01 = np.linspace(0, 1, P.shape[0])
plotit = 1
freq = 5
spheroid_gif = []
print("rayon", R)

for nt in range(0, len(t), freq):
    Pint = np.copy(P[:, nt])
    f = interpolate.interp1d(R[nt] * r01, r01, fill_value='extrapolate')
    Pint = f(P[:, nt])
    Pint[np.isnan(Pint)] = 0
    Pr = np.concatenate((np.flip(Pint), Pint)).reshape((1, -1))

    r = np.hstack((-np.flip(r01), r01)).reshape((-1, 1))
    # print("r", len(r))
    gridx, gridy = np.meshgrid(r, r)

    dist = np.sqrt(gridx ** 2 + gridy ** 2)
    A = np.zeros((len(gridx), len(gridy)))
    for k in range(len(r01)):
        if r[k] <= R[nt]:
            D = (dist > r01[k] - (r[2] - r[1])) & (dist < r01[k] + (r[2] - r[1]))
            A[D] = Pint[k]
    D = (dist > R[nt] - (r[2] - r[1])) & (dist < R[nt] + (r[2] - r[1]))
    # A[D] = 1
    # A[A == 0] = 1
    spheroid_gif.append(255 - A)

    fig = plt.gcf()
    plt.imshow(255 - A, cmap='gray')
    plt.title('t = {:.3f}h'.format(t[nt]))
    fig.set_dpi(100)
    fig.set_size_inches(1.5, 1.5)
    fig.savefig('Results/Radius{}.pdf'.format(plotit), bbox_inches='tight')
    plt.close(fig)
    plotit += 1

imageio.mimsave('Results/spheroid_evolution.gif', spheroid_gif)
