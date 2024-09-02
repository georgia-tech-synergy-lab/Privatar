import numpy as np 

mi_list = [0.01, 0.1, 1]
for v in mi_list:
    noise_cov_path = f"/home/jianming/work/Privatar_prj/experiment_scripts/pac_analysis/noise_covariance/noise_sigma_ibdct_decoder_0_{v}.npy"
    noise_cov = np.load(noise_cov_path)
    print(np.linalg.norm(noise_cov))