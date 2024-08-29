import numpy as np
from tqdm import tqdm
import torch 
import os

# Example: Generating a random dataset with 14000 samples, each with 256 features
# In a real scenario, replace this with your actual data loading
number_files = 5566
batch_size = 8

outsource_freq_list = [14]
# original model -- 5566 (8)
### --- The first one is 2*256, but all things following it is 8*256

# horizontal_partition_4 -- 9073 (batch_size)
# original_model -- 5566 (batch_size)
### --- mean 8*256
### --- std 8*256

# horizontal_partition_5 -- 9073 (batch_size)
### --- mean 8*256
### --- std 8*256

is_calcualted = False
if is_calcualted == False:
    for threshold in outsource_freq_list:
        captured_data_list = f"/home/jianming/work/Privatar_prj/testing_results/test_test_bdct4x4_hp_nn_decode_{threshold}/latent_code"
        # captured_data_list = f"/storage/ice1/3/0/jtong45/Privatar/testing_results/test_bdct4x4_hp_nn_decode_{threshold}/latent_code"
        captured_z_data = np.zeros(((number_files-1)*batch_size, 256))
        captured_z_outsource_data = np.zeros(((number_files-1)*batch_size, 256))

        for i in tqdm(range(number_files-1)):
            z_file_list = f"{captured_data_list}/z_{i+1}.pth"
            captured_z = torch.load(z_file_list).to("cpu")

            captured_z_data[i*batch_size:(i+1)*batch_size] = captured_z.detach().numpy()

        covariance_matrix_z = np.cov(captured_z_data, rowvar=False)

        # Print the covariance matrix
        print("Covariance Matrix outsourced:")
        print(covariance_matrix_z)
        np.save(f"covariance_matrix_{threshold}_z.npy", covariance_matrix_z)

        print(np.linalg.det(covariance_matrix_z))

        for i in tqdm(range(number_files-1)):
            z_outsource_file_list = f"{captured_data_list}/z_outsource_{i+1}.pth"
            captured_z_outsource = torch.load(z_outsource_file_list).to("cpu")

            captured_z_outsource_data[i*batch_size:(i+1)*batch_size] = captured_z_outsource.detach().numpy()

        covariance_matrix_z_outsource = np.cov(captured_z_outsource_data, rowvar=False)

        # Print the covariance matrix
        print("Covariance Matrix outsourced:")
        print(covariance_matrix_z_outsource)
        np.save(f"covariance_matrix_{threshold}_z_outsource.npy", covariance_matrix_z_outsource)

        print(np.linalg.det(covariance_matrix_z_outsource))

        # captured_logstd_data  = np.zeros((number_files*batch_size, 256))
        # captured_mean_data = np.zeros((number_files*batch_size, 256))
        # captured_z_data = np.zeros((number_files*batch_size, 256))
        # # captured_kl_data  = np.zeros((number_files*batch_size, 256))

        # for i in range(number_files):
        #     logstd_file_list = f"/home/jianming/work/multiface/captured_distribution_{threshold}/logstd_{i}.pth"
        #     captured_logstd = torch.load(logstd_file_list).to("cpu")
        #     mean_file_list = f"/home/jianming/work/multiface/captured_distribution_{threshold}/mean_{i}.pth"
        #     captured_mean = torch.load(mean_file_list).to("cpu")
        #     z_file_list = f"/home/jianming/work/multiface/captured_distribution_{threshold}/z_{i}.pth"
        #     captured_z = torch.load(z_file_list).to("cpu")
        #     # kl_file_list = f"/home/jianming/work/multiface/captured_distribution_{threshold}/kl_{i}.pth"
        #     # captured_kl = torch.load(kl_file_list).to("cpu")

        #     captured_logstd_data[i*batch_size:(i+1)*batch_size] = captured_logstd.detach().numpy()
        #     captured_mean_data[i*batch_size:(i+1)*batch_size] = captured_mean.detach().numpy()
        #     captured_z_data[i*batch_size:(i+1)*batch_size] = captured_z.detach().numpy()
        #     # captured_kl_data[i*batch_size:(i+1)*batch_size] =  captured_kl.detach().numpy()

        # # Calculate the covariance matrix
        # covariance_matrix_logstd = np.cov(captured_logstd_data, rowvar=False)
        # covariance_matrix_mean = np.cov(captured_mean_data, rowvar=False)
        # covariance_matrix_z = np.cov(captured_z_data, rowvar=False)
        # # covariance_matrix_kl = np.cov(captured_kl_data, rowvar=False)

        # # Print the covariance matrix
        # print("Covariance Matrix:")
        # print(covariance_matrix_logstd)
        # print(covariance_matrix_mean)
        # print(covariance_matrix_z)
        # # print(covariance_matrix_kl)

        # np.save(f"covariance_matrix_{threshold}_logstd.npy", covariance_matrix_logstd)
        # np.save(f"covariance_matrix_{threshold}_mean.npy", covariance_matrix_mean)
        # np.save(f"covariance_matrix_{threshold}_z.npy", covariance_matrix_z)
        # # np.save(f"covariance_matrix_{threshold}_kl.npy", covariance_matrix_kl)

        # print(np.linalg.det(covariance_matrix_logstd))
        # print(np.linalg.det(covariance_matrix_mean))
        # print(np.linalg.det(covariance_matrix_z))
        # # print(np.linalg.det(covariance_matrix_kl))
else:
    for threshold in outsource_freq_list:
        covariance_matrix_z_outsource = np.load(f"covariance_matrix_{threshold}_z_outsource.npy")

        unit_diagonal = np.diag(np.full(covariance_matrix_z_outsource.shape[0],-2))
        
        U, s3, V = np.linalg.svd(covariance_matrix_z_outsource)
        print("outsourced data")

        covariance_matrix_z = np.load(f"covariance_matrix_{threshold}_z.npy")


        unit_diagonal = np.diag(np.full(covariance_matrix_z.shape[0],-2))
        
        _, s3_typical, _ = np.linalg.svd(covariance_matrix_z)
        print("normal data")
