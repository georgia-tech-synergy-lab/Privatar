import os

# Please select one path out of the following three paths based on the server
path_prefix_god2 ="/home/jianming/work/multiface/"
path_prefix_synergy3 = "/home/jianming/work/multiface/"

path_testing_latent_code_god2 = "/home/jianming/work/Privatar_prj/testing_results/noisy_original_model_isotropic_noise_246.64_mutual_info_1_"
path_testing_latent_code_synergy3 = "/usr/scratch/jianming/Privatar/testing_results/noisy_original_model_isotropic_noise_246.64_mutual_info_1_"
average_texture_path = "dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"

best_model_path = "/home/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth"
# best_model_path = "/home/jianming/work/Privatar_prj/training_results/noisy_horizontal_partition_0_isotropic_noise_2450_mutual_info_0.1/best_model.pth"

#######
path_prefix = path_prefix_god2
path_testing_latent_code = path_testing_latent_code_god2
path_variance_matrix_tensor = "/home/jianming/work/Privatar_prj/profiled_latent_code/statistics/isotropic_noise_2450_mutual_information_0.1.pth"
# path_variance_matrix_tensor = "/home/jianming/work/Privatar_prj/profiled_latent_code/statistics/isotropic_noise_246.46_mutual_information_1.pth"
prefix_path_captured_latent_code = "/home/jianming/work/Privatar_prj/testing_results/dummy_path"
#######

BDCT_threshold = [0] 
# BDCT_threshold = [0.4, 0.8, 1, 1.6, 2.4, 3, 4, 5, 6, 19] 

# On syenrgy3 machine --- the following codes should be executed
# scl enable devtoolset-11 bash
#  --master_port=25678
for BDCT_thres in BDCT_threshold:
    result_path = f"{path_testing_latent_code}{str(BDCT_thres)}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print(f'torchrun --nproc_per_node=1 --master_port=25672  test.py --data_dir {path_prefix}/dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {path_prefix}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path} --frequency_threshold {BDCT_thres} --arch base --average_texture_path {path_prefix}{average_texture_path} --apply_gaussian_noise --prefix_path_captured_latent_code {prefix_path_captured_latent_code} --path_variance_matrix_tensor {path_variance_matrix_tensor}')
    os.system(f'torchrun --nproc_per_node=1 --master_port=25672  test.py --data_dir {path_prefix}/dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {path_prefix}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path} --frequency_threshold {BDCT_thres} --arch base --average_texture_path {path_prefix}{average_texture_path}  --apply_gaussian_noise --prefix_path_captured_latent_code {prefix_path_captured_latent_code} --path_variance_matrix_tensor {path_variance_matrix_tensor}')
