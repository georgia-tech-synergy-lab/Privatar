import numpy as np 

mi_list = [0.01, 0.1, 1]
for v in mi_list:
    noise_cov_path = f"/workspace/uwing2/Privatar/experiment_scripts/pac_analysis/noise_covariance/noise_sigma_ibdct_decoder_2_{v}.npy"
    noise_cov = np.load(noise_cov_path)
    print(np.linalg.norm(noise_cov))