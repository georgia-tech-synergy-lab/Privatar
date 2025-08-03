import numpy as np
from tqdm import tqdm
import torch 
import math
import os
import re

file_dump_name_suffix = "_ibdct_decoder"
path_to_privatar = "/scratch2/jianming/work/Privatar_prj"
outsource_freq_list = [2,4,6,8,10,12,14,16]
mutual_info_bound_list = [4, 3, 1, 0.1, 0.01]
posterious_successful_rate_list = [0.98, 0.827, 0.4, 0.09, 0.035] # corresponding to `mutual_info_bound_list = [4, 3, 1, 0.1, 0.01]`
prior_successful_rate = 1 / 56 # In total 56 expressions under expression identification attack.
dimensionality = 256

#########################
# Path checking.
if not os.path.exists("./l2norm"):
  os.makedirs("./l2norm")
if not os.path.exists("./generated_dp_noise"):
  os.makedirs("./generated_dp_noise")
#########################

#########################
### Predefined Function
def get_max_index(folder_path):
  # List all files in the directory
  files = os.listdir(folder_path)
  
  # Regular expression to match the pattern z_<number>.pth
  pattern = re.compile(r'z_outsource_(\d+)\.pth')
  
  # Extract all numbers from the file names
  indices = [int(pattern.search(f).group(1)) for f in files if pattern.match(f)]
  
  # Return the maximum index (N)
  return max(indices) if indices else None


def dp_noise_calculation(l2_norm_outsourced_data, prior_successful_rate, posterious_successful_rate_list):
  """
  [1] https://arxiv.org/pdf/1702.07476**

  First, calculate \( \epsilon = \frac{\alpha}{2 \sigma^2} \).

  Where \( y = \text{Posterior Successful Rate (PSR)} \),

  \( k = \text{Prior Successful Rate} \) (under the expression identification attack, it is \( \frac{1}{65} \)).

  In Theorem 2, \( \epsilon = \frac{\alpha c^2}{2 \sigma^2} \), where \( c^2 = \text{profiled \( L_2 \) norm (260)} \).

  Substitute \( \epsilon \), \( y \), and \( k \) into Theorem 2 to obtain an equation involving \( \alpha \) and \( \sigma \).

  In this equation, iterate through \( \alpha = 1 \) to \( 20 \), calculate \( \sigma \), and select the smallest value of \( \sigma \).

  sigma^2 = alpha*c^2/[2*alpha/(alpha-1)*ln{pri} - 2*ln{PSR}]

  For the PAC experiment, we used the following configurations:

  1. **MI = 1 (PSR = 0.40)**  
  2. **MI = 0.1 (PSR = 0.09)**  
  3. **MI = 0.01 (PSR = 0.035)**  
  """

  c_square = l2_norm_outsourced_data * l2_norm_outsourced_data
  kappa = prior_successful_rate
  result_sigma_list = []
  for gamma in posterious_successful_rate_list:
    minimal_sigma = 9999999999
    for alpha in range(2,20):
      divisor = (2*alpha/(alpha-1)*math.log(gamma) - 2*math.log(kappa))
      if divisor > 0:
        sigma_square = alpha*c_square/divisor
        if sigma_square < minimal_sigma:
          minimal_sigma = sigma_square
    result_sigma_list.append(minimal_sigma)

  return result_sigma_list
###########################


local_branch_noise_covariance_by_freqnum_by_noise = np.zeros((len(outsource_freq_list), len(posterious_successful_rate_list)))
outsource_branch_noise_covariance_by_freqnum_by_noise = np.zeros((len(outsource_freq_list), len(posterious_successful_rate_list)))
# Specify your folder path
overall_noise_trace_list = []


for j, outsource_freq_num in enumerate(outsource_freq_list):

  #########################
  ### Please modify!!
  if outsource_freq_num==16:
    folder_path = f'{path_to_privatar}/testing_results/test_multiface_direct_split/latent_code'
  else:
    folder_path = f'{path_to_privatar}/testing_results/test_bdct_hp_ibdct_decoder_{outsource_freq_num}/latent_code'
  #########################

  #########################
  # Read external files
  #########################
  captured_data_list = folder_path
  N = get_max_index(folder_path)
  print(f'The value of N (number_files) is: {N}')
  number_files = N

  first_latent_code = torch.load(os.path.join(folder_path, "z_outsource_0.pth"))
  last_latent_code = torch.load(os.path.join(folder_path, f"z_outsource_{N}.pth"))
  major_latent_code = torch.load(os.path.join(folder_path, "z_outsource_2.pth"))
  batch_size = major_latent_code.shape[0]
  dim_size = major_latent_code.shape[1]
  print(f'The batch size list: {first_latent_code.shape[0]}, {batch_size}, {last_latent_code.shape[0]} for the first, major and the last latent code')
  total_test_size =  (N-2) * batch_size + first_latent_code.shape[0] + last_latent_code.shape[0]
  print(f'Overall data shape = ({total_test_size}, {dim_size})')

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
  
  sigma_under_various_psr = dp_noise_calculation(l2_norm_outsourced, prior_successful_rate, posterious_successful_rate_list)
  sgl_outsource_noise_trace_list = []
  for i, sigma in enumerate(sigma_under_various_psr):
    trace_noise = np.sum(math.sqrt(sigma)*dimensionality)
    print(f"when num_freq_comp_outsourced={outsource_freq_num}, Accumulation of simga_i (trace) of noise covariance = {trace_noise} when MI = {mutual_info_bound_list[i]}")
    # print(f"when num_freq_comp_outsourced={outsource_freq_num}, Sigma of DP noise = {math.sqrt(sigma)} under psr={posterious_successful_rate_list[i]}")
    dp_noise = np.ones(dimensionality) * math.sqrt(sigma)
    np.save(f"./generated_dp_noise/dp_noise{file_dump_name_suffix}_{outsource_freq_num}_{mutual_info_bound_list[i]}.npy", dp_noise)
    sgl_outsource_noise_trace_list.append(trace_noise)
  overall_noise_trace_list.append(sgl_outsource_noise_trace_list)

for i in range(len(overall_noise_trace_list[0])):
  print(f"mi = {mutual_info_bound_list[i]}, noise trace = ", end=" ")
  for j in range(len(overall_noise_trace_list)):
    print(f"{overall_noise_trace_list[j][i]}", end = " ")
print("FINISH!")
