import numpy as np
from tqdm import tqdm
import torch 
import os
import re

file_dump_name_suffix = "_ibdct_decoder"
outsource_freq_list = [2,4,6,8,10,12,14]
mutual_info_bound_list = [1, 0.1, 0.01]

calculate_dp_noise = True

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
local_branch_trace_latent_code_covariance = []
outsource_branch_trace_latent_code_covariance = []
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

    l2_norm = np.linalg.norm(captured_z_data)
    # Print the covariance matrix
    np.save(f"./l2norm/l2norm_local_branch_when_outsource_freq_num_{outsource_freq_num}_z.npy", l2_norm)
    print(f"l2norm of local vector when outsource_freq_num={outsource_freq_num}_z.npy", l2_norm)


    ##############################
    # Outsourced Branch
    ##############################
    captured_z_outsource_data = np.zeros((total_test_size, dim_size))
    if outsource_freq_num > 0:
        for k in tqdm(range(number_files)):
            z_outsource_file_list = f"{captured_data_list}/z_outsource_{k+1}.pth"
            captured_z_outsource = torch.load(z_outsource_file_list).to("cpu")

            captured_z_outsource_data[k*batch_size:(k+1)*batch_size] = captured_z_outsource.detach().numpy()

        l2_norm_outsourced = np.linalg.norm(captured_z_outsource_data)
        
        # Print the covariance matrix
        np.save(f"./l2norm/l2norm_outsource_freq_num_{outsource_freq_num}_z.npy", l2_norm_outsourced)
        print(f"l2norm of outsourced vector when outsource_freq_num={outsource_freq_num}_z.npy", l2_norm_outsourced)
