import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Parameters ---
N = 256  # number of dimensions for the radar plot

# Example data: replace this with your own 256-element array if desired
values = torch.load("/usr/scratch/jtong45/Privatar/testing_result/test_bdct_hp_ibdct_decoder_2/latent_code/z_39.pth")  # one value per dimension in [0, 1]
noise_dp = np.load("/usr/scratch/jtong45/Privatar/experiment_scripts/dp_analysis/generated_dp_noise/dp_noise_ibdct_decoder_2_1.npy")
noise_pac = np.load("/usr/scratch/jtong45/Privatar/experiment_scripts/pac_analysis/noise_covariance/noise_sigma_outsource_ibdct_decoder_2_1.npy")
values = values[0,...].to("cpu").detach().numpy()
gaussian_noise_covariance_dp  = np.diag(noise_dp)
gaussian_noise_covariance_pac = np.diag(noise_pac)
mean = np.zeros(gaussian_noise_covariance_dp.shape[0]) 
samples_dp = np.random.multivariate_normal(mean, gaussian_noise_covariance_dp, 1).reshape(-1)
samples_pac = np.random.multivariate_normal(mean, gaussian_noise_covariance_pac, 1).reshape(-1)
# --- Radar plot helper ---
# Angles for each axis, in radians
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

# Close the loop by appending the first element to the end
theta = np.concatenate([theta, theta[:1]])
values = np.concatenate([values, values[:1]])
noise_dp = np.concatenate([noise_dp, noise_dp[:1]])
noise_pac = np.concatenate([noise_pac, noise_pac[:1]])
samples_dp = np.concatenate([samples_dp, samples_dp[:1]])
samples_pac = np.concatenate([samples_pac, samples_pac[:1]])

# --- Plot ---
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)

ax.plot(theta, values)          # outline
ax.fill(theta, values, alpha=0.25)  # lightly fill the area
ax.plot(theta, samples_dp)
ax.plot(theta, samples_pac)
BIGGER_SIZE = 16

# Tidy up: show a subset of axis labels to avoid clutter
subset = 16  # show only 16 evenly spaced labels
ax.set_xticks(theta[:-1: N // subset], labels=[f"{i*N//subset}" for i in range(subset)], fontsize=BIGGER_SIZE)
#ax.set_yticklabels([])  # hide radial labels for clarity

# ax.set_title(f"Example {N}-Dimensional Radar Plot", pad=20)
# ax.fill(theta, values_closed, alpha=0.25, zorder=1)
# Minimal tick labels to avoid clutter
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels

plt.tight_layout()

plt.savefig('noisy_value_visualization_dp.pdf', bbox_inches="tight", transparent=True)
