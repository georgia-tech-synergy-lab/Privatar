import os

# Please select one path out of the following three paths based on the server
data_dir = "/workspace/uwing2/multiface"
result_path_prefix = "/workspace/uwing2/Privatar/testing_results/"

val_batch_size = 24
num_freq_comp_outsourced = 2
project_name = f"noisy_test_bdct_hp_ibdct_decoder_0{num_freq_comp_outsourced}"
best_model_path = f"/workspace/uwing2/Privatar/training_results/bdct_hp_nn_decoder_{num_freq_comp_outsourced}/best_model.pth"
result_path = f"{result_path_prefix}{project_name}"

mi_list = [0.01, 0.1, 1]
for v in mi_list:
    gaussian_noise_covariance_path = f"/home/jianming/work/Privatar_prj/custom_scripts/pac_analysis/noise_covariance/noise_sigma_ibdct_decoder_0_{v}.npy"
        
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print(f'python test_sgl_run_uwing2.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --num_freq_comp_outsourced {num_freq_comp_outsourced}   --gaussian_noise_covariance_path {gaussian_noise_covariance_path}')
    os.system(f'python test_sgl_run_uwing2.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}   --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --num_freq_comp_outsourced {num_freq_comp_outsourced}   --gaussian_noise_covariance_path {gaussian_noise_covariance_path}')
