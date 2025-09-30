import numpy as np
from tqdm import tqdm
import torch 
import os
import re

file_dump_name_suffix = "_decoder"
path_to_privatar = "/work"
offload_freq_list = [2,4,6,8,10,12,14,16]
mutual_info_bound_list = [4, 3, 1, 0.1, 0.01]
posterious_successful_rate_list = [0.98, 0.827, 0.4, 0.09, 0.035] # corresponding to `mutual_info_bound_list = [4, 3, 1, 0.1, 0.01]`

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
    pattern = re.compile(r'z_offload_(\d+)\.pth')
    
    # Extract all numbers from the file names
    indices = [int(pattern.search(f).group(1)) for f in files if pattern.match(f)]
    
    # Return the maximum index (N)
    return max(indices) if indices else None
###########################

local_branch_noise_covariance_by_freqnum_by_noise = np.zeros((len(offload_freq_list), len(mutual_info_bound_list)))
offload_branch_noise_covariance_by_freqnum_by_noise = np.zeros((len(offload_freq_list), len(mutual_info_bound_list)))
# Specify your folder path
local_branch_l2_norm_latent_code_covariance = []
offload_branch_l2_norm_latent_code_covariance = []
local_branch_trace_latent_code_covariance = []
offload_branch_trace_latent_code_covariance = []

for j, offload_freq_num in enumerate(offload_freq_list):

    #########################
    ### Please modify!!
    if offload_freq_num==16:
        folder_path = f'{path_to_privatar}/testing_results/test_multiface_direct_split/latent_code'
    else:
        folder_path = f'{path_to_privatar}/testing_results/test_partition_{offload_freq_num}/latent_code'
  
    #########################

    #########################
    # Read external files
    #########################
    captured_data_list = folder_path
    N = get_max_index(folder_path)
    print(f'The value of N (number_files) is: {N}')
    number_files = N

    first_latent_code = torch.load(os.path.join(folder_path, "z_offload_0.pth"))
    last_latent_code = torch.load(os.path.join(folder_path, f"z_offload_{N}.pth"))
    major_latent_code = torch.load(os.path.join(folder_path, "z_offload_2.pth"))
    batch_size = major_latent_code.shape[0]
    dim_size = major_latent_code.shape[1]
    print(f'The batch size list: {first_latent_code.shape[0]}, {batch_size}, {last_latent_code.shape[0]} for the first, major and the last latent code')
    total_test_size =  (N-2) * batch_size + first_latent_code.shape[0] + last_latent_code.shape[0]
    print(f'Overall data shape = ({total_test_size}, {dim_size})')

    ##############################
    # local Branch
    ##############################
    if offload_freq_num<16:
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
        np.save(f"./covariance/covariance_matrix_partition_local{file_dump_name_suffix}_{offload_freq_num}_z.npy", covariance_matrix_z)

        print(f"L2 norm of covariance matrix = {np.linalg.norm(covariance_matrix_z)}")
        local_branch_l2_norm_latent_code_covariance.append(np.linalg.norm(covariance_matrix_z))
        print(f"Trace of covariance matrix = {np.trace(covariance_matrix_z)}")
        local_branch_trace_latent_code_covariance.append(np.trace(covariance_matrix_z))
        unit_diagonal = np.diag(np.full(covariance_matrix_z.shape[0],-2))
        
        # SVD Decomposition
        u, s, u_t = np.linalg.svd(covariance_matrix_z)
        np.save(f"./svd/u_partition_local{file_dump_name_suffix}_{offload_freq_num}.npy", u)
        np.save(f"./svd/s_partition_local{file_dump_name_suffix}_{offload_freq_num}.npy", s)
        np.save(f"./svd/u_t_partition_local{file_dump_name_suffix}_{offload_freq_num}.npy", u_t)

        # Noise Covariance Calculation and Injection.
        acc_s = np.sum(np.sqrt(s))
        for i, v in enumerate(mutual_info_bound_list):
            noise_s = np.empty_like(s)
            for dim_id in range(noise_s.shape[0]):
                noise_s[dim_id] = s[dim_id] * acc_s / 2 / v
            np.save(f"./noise_covariance/pac_noise_partition_local{file_dump_name_suffix}_{offload_freq_num}_{v}.npy", noise_s)
            print(f"when num_freq_comp_offloadd={offload_freq_num}, Accumulation of simga_i (trace) of noise covariance = {np.sum(noise_s)} when MI = {v}")
            local_branch_noise_covariance_by_freqnum_by_noise[j,i] = np.sum(noise_s)

    ##############################
    # offloadd Branch
    ##############################
    captured_z_offload_data = np.zeros((total_test_size, dim_size))
    if offload_freq_num > 0:
        for k in tqdm(range(number_files)):
            z_offload_file_list = f"{captured_data_list}/z_offload_{k+1}.pth"
            captured_z_offload = torch.load(z_offload_file_list).to("cpu")

            captured_z_offload_data[k*batch_size:(k+1)*batch_size] = captured_z_offload.detach().numpy()

        covariance_matrix_z_offload = np.cov(captured_z_offload_data, rowvar=False)

        # Print the covariance matrix
        print("Covariance Matrix -- offloadd latent code:")
        print(covariance_matrix_z_offload)
        np.save(f"./covariance/covariance_matrix_partition_offloaded{file_dump_name_suffix}_{offload_freq_num}_z_offload.npy", covariance_matrix_z_offload)

        print(f"L2 norm of covariance matrix = {np.linalg.norm(covariance_matrix_z_offload)}")
        offload_branch_l2_norm_latent_code_covariance.append(np.linalg.norm(covariance_matrix_z_offload))
        print(f"trace of covariance matrix = {np.trace(covariance_matrix_z_offload)}")
        offload_branch_trace_latent_code_covariance.append(np.trace(covariance_matrix_z_offload))
        u_offload, s_offload, u_t_offload = np.linalg.svd(covariance_matrix_z_offload)
        np.save(f"./svd/u_partition_offloaded{file_dump_name_suffix}_{offload_freq_num}.npy", u_offload)
        np.save(f"./svd/s_partition_offloaded{file_dump_name_suffix}_{offload_freq_num}.npy", s_offload)
        np.save(f"./svd/u_t_partition_offloaded{file_dump_name_suffix}_{offload_freq_num}.npy", u_t_offload)

        # Noise Covariance Calculation and Injection.
        acc_s_offload = np.sum(np.sqrt(s_offload))
        for i, v in enumerate(mutual_info_bound_list):
            noise_s_offload = np.empty_like(s_offload)
            for dim_id in range(noise_s_offload.shape[0]):
                noise_s_offload[dim_id] = s_offload[dim_id] * acc_s_offload / 2 / v
            np.save(f"./noise_covariance/pac_noise_partition_offloaded{file_dump_name_suffix}_{offload_freq_num}_{v}.npy", noise_s_offload)
            print(f"when num_freq_comp_offloadd={offload_freq_num}, Accumulation of simga_i (trace) of offloadd noise covariance = {np.sum(noise_s_offload)} when MI = {v}")
            offload_branch_noise_covariance_by_freqnum_by_noise[j,i] = np.sum(noise_s_offload)

print(f"local l2 norm of covariance for local latent code matrix")
print(local_branch_l2_norm_latent_code_covariance)

print(f"trace of covariance of local latent code matrix")
print(local_branch_trace_latent_code_covariance)

print(f"local noise covariance (vertical -- diff. freq. component; horizontal -- diff. mutual information)")
for j in range(len(offload_freq_list)):
    for i in range(len(mutual_info_bound_list)):
        print(local_branch_noise_covariance_by_freqnum_by_noise[j,i],end=" ")
    print()

print(f"offloadd l2 norm of covariance for offloadd latent code matrix")
print(offload_branch_l2_norm_latent_code_covariance)

print(f"offloadd trace of covariance for offloadd latent code matrix")
print(offload_branch_trace_latent_code_covariance)
print(f"offloadd noise covariance (vertical -- diff. freq. component; horizontal -- diff. mutual information)")
for j in range(len(offload_freq_list)):
    for i in range(len(mutual_info_bound_list)):
        print(offload_branch_noise_covariance_by_freqnum_by_noise[j,i],end=" ")
    print()

print("FINISH!")