import numpy as np
from tqdm import tqdm
import torch 
import os
import re

file_dump_name_suffix = "_ibdct_decoder"
outsource_freq_list = [2,4,6,8,10,12,14]
mutual_info_bound_list = [1, 0.1, 0.01]

#########################
# Path checking.
if not os.path.exists("./noise_covariance"):
    os.makedirs("./noise_covariance")
if not os.path.exists("./covariance"):
    os.makedirs("./covariance")
if not os.path.exists("./svd"):
    os.makedirs("./svd")
#########################

#########################
### Predefined Function
def get_max_index(folder_path):
    # List all files in the directory
    files = os.listdir(folder_path)
    
    # Regular expression to match the pattern z_<number>.pth
    pattern = re.compile(r'z_(\d+)\.pth')
    
    # Extract all numbers from the file names
    indices = [int(pattern.search(f).group(1)) for f in files if pattern.match(f)]
    
    # Return the maximum index (N)
    return max(indices) if indices else None
###########################

local_branch_noise_covariance_by_freqnum_by_noise = np.zeros((len(outsource_freq_list), len(mutual_info_bound_list)))
outsource_branch_noise_covariance_by_freqnum_by_noise = np.zeros((len(outsource_freq_list), len(mutual_info_bound_list)))
# Specify your folder path
local_branch_l2_norm_latent_code_covariance = []
outsource_branch_l2_norm_latent_code_covariance = []
for j, outsource_freq_num in enumerate(outsource_freq_list):

    #########################
    ### Please modify!!
    folder_path = f'/workspace/uwing2/Privatar/testing_results/test_bdct_hp_ibdct_decoder_{outsource_freq_num}/latent_code'
    #########################

    #########################
    # Read external files
    #########################
    captured_data_list = folder_path
    N = get_max_index(folder_path)
    print(f'The value of N (number_files) is: {N}')
    number_files = N

    first_latent_code = torch.load(os.path.join(folder_path, "z_0.pth"))
    last_latent_code = torch.load(os.path.join(folder_path, f"z_{N}.pth"))
    major_latent_code = torch.load(os.path.join(folder_path, "z_2.pth"))
    batch_size = major_latent_code.shape[0]
    dim_size = major_latent_code.shape[1]
    print(f'The batch size list: {first_latent_code.shape[0]}, {batch_size}, {last_latent_code.shape[0]} for the first, major and the last latent code')
    total_test_size =  (N-2) * batch_size + first_latent_code.shape[0] + last_latent_code.shape[0]
    print(f'Overall data shape = ({total_test_size}, {dim_size})')

    ##############################
    # local Branch
    ##############################
    captured_z_data = np.zeros((total_test_size, dim_size))

    # Calculate Covariance
    for k in tqdm(range(number_files)):
        z_file_list = f"{captured_data_list}/z_{k+1}.pth"
        captured_z = torch.load(z_file_list).to("cpu")

        captured_z_data[k*batch_size:(k+1)*batch_size] = captured_z.detach().numpy()

    covariance_matrix_z = np.cov(captured_z_data, rowvar=False)

    # Print the covariance matrix
    print("Covariance Matrix -- local latent code:")
    print(covariance_matrix_z)
    np.save(f"./covariance/covariance_matrix{file_dump_name_suffix}_{outsource_freq_num}_z.npy", covariance_matrix_z)

    print(f"L2 norm of covariance matrix = {np.linalg.norm(covariance_matrix_z)}")
    local_branch_l2_norm_latent_code_covariance.append(np.linalg.norm(covariance_matrix_z))
    unit_diagonal = np.diag(np.full(covariance_matrix_z.shape[0],-2))
    
    # SVD Decomposition
    u, s, u_t = np.linalg.svd(covariance_matrix_z)
    np.save(f"./svd/u{file_dump_name_suffix}_{outsource_freq_num}.npy", u)
    np.save(f"./svd/s{file_dump_name_suffix}_{outsource_freq_num}.npy", s)
    np.save(f"./svd/u_t{file_dump_name_suffix}_{outsource_freq_num}.npy", u_t)

    # Noise Covariance Calculation and Injection.
    acc_s = np.sum(np.sqrt(s))
    for i, v in enumerate(mutual_info_bound_list):
        noise_s = np.empty_like(s)
        for dim_id in range(noise_s.shape[0]):
            noise_s[dim_id] = s[dim_id] * acc_s / 2 / v
        np.save(f"./noise_covariance/noise_sigma{file_dump_name_suffix}_{outsource_freq_num}_{v}.npy", noise_s)
        print(f"L2 norm of noise covariance = {np.linalg.norm(noise_s)} when MI = {v}")
        local_branch_noise_covariance_by_freqnum_by_noise[j,i] = np.linalg.norm(noise_s)

    ##############################
    # Outsourced Branch
    ##############################
    captured_z_outsource_data = np.zeros((total_test_size, dim_size))
    if outsource_freq_num > 0:
        for k in tqdm(range(number_files)):
            z_outsource_file_list = f"{captured_data_list}/z_outsource_{k+1}.pth"
            captured_z_outsource = torch.load(z_outsource_file_list).to("cpu")

            captured_z_outsource_data[k*batch_size:(k+1)*batch_size] = captured_z_outsource.detach().numpy()

        covariance_matrix_z_outsource = np.cov(captured_z_outsource_data, rowvar=False)

        # Print the covariance matrix
        print("Covariance Matrix -- outsourced latent code:")
        print(covariance_matrix_z_outsource)
        np.save(f"./covariance/covariance_matrix{file_dump_name_suffix}_{outsource_freq_num}_z_outsource.npy", covariance_matrix_z_outsource)

        print(f"L2 norm of covariance matrix = {np.linalg.norm(covariance_matrix_z_outsource)}")
        outsource_branch_l2_norm_latent_code_covariance.append(np.linalg.norm(covariance_matrix_z_outsource))
        u_outsource, s_outsource, u_t_outsource = np.linalg.svd(covariance_matrix_z_outsource)
        np.save(f"./svd/u_outsource{file_dump_name_suffix}_{outsource_freq_num}.npy", u_outsource)
        np.save(f"./svd/s_outsource{file_dump_name_suffix}_{outsource_freq_num}.npy", s_outsource)
        np.save(f"./svd/u_t_outsource{file_dump_name_suffix}_{outsource_freq_num}.npy", u_t_outsource)

        # Noise Covariance Calculation and Injection.
        acc_s_outsource = np.sum(np.sqrt(s_outsource))
        for i, v in enumerate(mutual_info_bound_list):
            noise_s_outsource = np.empty_like(s)
            for dim_id in range(noise_s_outsource.shape[0]):
                noise_s_outsource[dim_id] = s_outsource[dim_id] * acc_s_outsource / 2 / v
            np.save(f"./noise_covariance/noise_sigma_outsource{file_dump_name_suffix}_{outsource_freq_num}_{v}.npy", noise_s_outsource)
            print(f"L2 norm of outsourced noise covariance = {np.linalg.norm(noise_s_outsource)} when MI = {v}")
            outsource_branch_noise_covariance_by_freqnum_by_noise[j,i] = np.linalg.norm(noise_s_outsource)

print(f"local l2 norm of covariance for local latent code matrix")
print(local_branch_l2_norm_latent_code_covariance)

print(f"local noise covariance (vertical -- diff. freq. component; horizontal -- diff. mutual information)")
for j in range(len(outsource_freq_list)):
    for i in range(len(mutual_info_bound_list)):
        print(local_branch_noise_covariance_by_freqnum_by_noise[j,i],end=" ")
    print()

print(f"outsourced l2 norm of covariance for outsourced latent code matrix")
print(outsource_branch_l2_norm_latent_code_covariance)

print(f"outsourced noise covariance (vertical -- diff. freq. component; horizontal -- diff. mutual information)")
for j in range(len(outsource_freq_list)):
    for i in range(len(mutual_info_bound_list)):
        print(outsource_branch_noise_covariance_by_freqnum_by_noise[j,i],end=" ")
    print()
