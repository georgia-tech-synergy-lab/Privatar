import os

# Please select one path out of the following three paths based on the server
data_dir = "/scratch2/multiface/dataset"
result_path_prefix = "/scratch2/jianming/work/Privatar_prj/testing_results/"

mi_list = [1, 0.1, 0.01]
val_batch_size = 32
num_freq_comp_outsourced_list = [2, 4, 6, 8, 10, 12, 14]
for num_freq_comp_outsourced in num_freq_comp_outsourced_list:
  for mi in mi_list:
    project_name = f"test_noisy_accuracy_HP_all_expressions_{num_freq_comp_outsourced}_{mi}"
    gaussian_noise_covariance_path = f"/scratch2/jianming/work/Privatar_prj/experiment_scripts/pac_analysis/noise_covariance/noise_sigma_outsource_ibdct_decoder_{num_freq_comp_outsourced}_{mi}.npy"
    best_model_path = f"/scratch2/jianming/work/Privatar_prj/training_results/bdct_hp_ibdct_decoder_{num_freq_comp_outsourced}/best_model.pth"
    result_path = f"{result_path_prefix}{project_name}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print(f'python test_accuracy.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size}  --num_freq_comp_outsourced {num_freq_comp_outsourced} --gaussian_noise_covariance_path {gaussian_noise_covariance_path}')
    os.system(f'python test_accuracy.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size}  --num_freq_comp_outsourced {num_freq_comp_outsourced} --gaussian_noise_covariance_path {gaussian_noise_covariance_path}')
