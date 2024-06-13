import os
import cv2
import math
import torch
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings

# perform noise calculation on the convariance matrix
def generation_l2norm_l2cov(z_data):
    covariance_matrix_z = np.cov(z_data, rowvar=False)
    l2_norm_cov = np.linalg.norm(covariance_matrix_z)
    U_cov, s_cov, V_cov = np.linalg.svd(covariance_matrix_z)

    overall_result_cov = 0
    for ele in s_cov:
        overall_result_cov = overall_result_cov + np.sqrt(ele)
    overall_result_cov = overall_result_cov / 256
    print(f"overall_result_cov={overall_result_cov}; s_cov={s_cov}")


    print("print detailed value of individual elements in convariance:")
    for ele in s_cov:
        print(ele, end=" ")
    return l2_norm_cov, U_cov, s_cov, V_cov

# calculate noise of each dimensions
def generate_noise_covariance(mutual_info_bound, s_cov):
    l2_norm_svd_decomposition_original = np.linalg.norm(s_cov)
    print(f"l2_norm_svd_decomposition_original = {l2_norm_svd_decomposition_original}")
    noise_variance = np.zeros(256)
    for i in range(256):
        noise_variance[i] =  2 * mutual_info_bound / (s_cov[i] * l2_norm_svd_decomposition_original)
    return noise_variance
    

# Function to sample from a 2D Gaussian distribution
def sample_2d_gaussian(mean, U_cov, noise_variance, V_cov, path_save_covariance, num_samples=1000):
    """
    Samples points from a 2D Gaussian distribution.

    Parameters:
    mean (array-like): Mean of the Gaussian distribution (shape: [256]).
    cov (array-like): Covariance matrix of the Gaussian distribution (shape: [256, 256]).
    num_samples (int): Number of samples to generate.

    Returns:
    np.ndarray: Samples from the Gaussian distribution (shape: [num_samples, 256]).
    """
    variance_matrix = (U_cov*noise_variance*V_cov)
    variance_matrix_tensor = torch.from_numpy(variance_matrix)
    torch.save(variance_matrix_tensor, path_save_covariance)
    samples = np.random.multivariate_normal(mean, variance_matrix, num_samples)
    return samples

def svd_decomposition_covariance_latent_code(covariance_matrix_z_outsource, path_save_covariance):
    U, s, V = np.linalg.svd(covariance_matrix_z_outsource)

    X = [i for i in range(covariance_matrix_z_outsource.shape[0])]
    Y = [1 for i in range(covariance_matrix_z_outsource.shape[0])]

    # Data for plotting
    SMALL_SIZE = 20
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 24

    plt.figure(figsize=(9, 6))
    plt.plot(X, s)
    plt.title('SVD decomposition of Cov(Z): (Z vector)', fontsize=SMALL_SIZE)

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.savefig(f"{path_save_covariance.split('.')[0]}_convariance.pdf", bbox_inches="tight", transparent=True) 


"""
    User Specified Input
"""


# original model -- 5566 (8)
### --- The first one is 2*256, but all things following it is 8*256

# horizontal_partition_4 -- 9073 (batch_size)
# original_model -- 5566 (batch_size)
### --- mean 8*256
### --- std 8*256

# horizontal_partition_5 -- 9073 (batch_size)
### --- mean 8*256
### --- std 8*256


threshold_list = [4]#4, 5]
number_files = 9073
batch_size = 8
num_element_mean = 256
mutual_info_bound = 0.1
num_samples = 1
path_save_covariance = f"/home/jianming/work/Privatar_prj/profiled_latent_code/noise_variance_matrix_horizontal_partition_{threshold_list[0]}_mutual_bound_{mutual_info_bound}_private_path_latent.pth"
captured_data_list = f"/home/jianming/work/Privatar_prj/testing_results/horizontal_partition_{threshold_list[0]}_latent_code"


if __name__ == "__main__":
    captured_z_data = np.zeros(((number_files-1)*batch_size, 256))

    for i in range(number_files-1):
        z_file_list = f"{captured_data_list}/z_{i+1}.pth"
        captured_z = torch.load(z_file_list).to("cpu")
        captured_z_data[i*batch_size:(i+1)*batch_size] = captured_z.detach().numpy()
    covariance_matrix_z = np.cov(captured_z_data, rowvar=False)
    svd_decomposition_covariance_latent_code(covariance_matrix_z, path_save_covariance)

    l2_norm_cov, U_cov, s_cov, V_cov = generation_l2norm_l2cov(captured_z_data)

    noise_variance = generate_noise_covariance(mutual_info_bound, s_cov)

    mean = np.zeros(num_element_mean)
    noise_samples = sample_2d_gaussian(mean, U_cov, noise_variance, V_cov, path_save_covariance, num_samples)
    print(noise_samples)
