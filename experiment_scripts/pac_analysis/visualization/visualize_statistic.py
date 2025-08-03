import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Parameters ---
N = 256  # number of dimensions for the radar plot

# Example data: replace this with your own 256-element array if desired
values = np.load("/usr/scratch/jtong45/Privatar/experiment_scripts/pac_analysis/svd/s_outsource_ibdct_decoder_2.npy")  # one value per dimension in [0, 1]
noise_pac = np.load("/usr/scratch/jtong45/Privatar/experiment_scripts/pac_analysis/noise_covariance/noise_sigma_outsource_ibdct_decoder_2_1.npy")
noise_dp = np.load("/usr/scratch/jtong45/Privatar/experiment_scripts/dp_analysis/generated_dp_noise/dp_noise_ibdct_decoder_2_1.npy")
noise_dp = np.log(noise_dp)
noise_pac = np.log(noise_pac)
# --- Radar plot helper ---
# Angles for each axis, in radians
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

# Close the loop by appending the first element to the end
theta = np.concatenate([theta, theta[:1]])
values = np.concatenate([values, values[:1]])
noise_pac = np.concatenate([noise_pac, noise_pac[:1]])
noise_dp = np.concatenate([noise_dp, noise_dp[:1]])

# --- Plot ---
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)

ax.plot(theta, values, label='trace of covariance')          # outline
ax.fill(theta, values, alpha=0.25)  # lightly fill the area
ax.plot(theta, noise_dp, label='trace of covariance (DP Noise)')
ax.plot(theta, noise_pac, label='trace of covariance (PAC Privacy)')
ax.legend(fontsize=14, loc='lower right')

BIGGER_SIZE = 16

# Tidy up: show a subset of axis labels to avoid clutter
subset = 16  # show only 16 evenly spaced labels
ax.set_xticks(theta[:-1: N // subset], labels=[f"{i*N//subset}" for i in range(subset)], fontsize=BIGGER_SIZE)
ax.set_yticklabels([r'$1$', r'$10$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$'], fontsize=12)  # set radial labels as powers of 10

# ax.set_title(f"Example {N}-Dimensional Radar Plot", pad=20)
# ax.fill(theta, values_closed, alpha=0.25, zorder=1)
# Minimal tick labels to avoid clutter
# ax.set_xticks(theta[:-1: N // subset], labels=[f"{i*N//subset}" for i in range(subset)])
# ax.set_yticklabels([])
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels

plt.tight_layout()

plt.savefig('pac_noise_visualization_value_N_noise.pdf', bbox_inches="tight", transparent=True)
