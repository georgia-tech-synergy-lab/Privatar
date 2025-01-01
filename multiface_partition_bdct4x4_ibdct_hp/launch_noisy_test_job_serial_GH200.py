import os

# Please select one path out of the following three paths based on the server
using_pac_noise = True # True for using PAC noise, False for using DP noise
path_to_privatar = "/workspace/uwing2/Privatar"
wandb_author_name = "jimmytong"

data_dir = f"{path_to_privatar}/dataset/m--20180227--0000--6795937--GHS" 

val_batch_size = 10
mi_list = [0.1, 0.01]
num_freq_comp_outsourced_list = [14] #[2, 4, 6, 8, 10, 12, 14]

for num_freq_comp_outsourced in num_freq_comp_outsourced_list:
    best_model_path = f"{path_to_privatar}/training_results/bdct_hp_ibdct_decoder_{num_freq_comp_outsourced}/best_model.pth"
    for v in mi_list:
        project_name = f"noisy_test_bdct_hp_ibdct_decoder_{num_freq_comp_outsourced}_{v}"
        result_path = f"{path_to_privatar}/testing_results/{project_name}"
        if using_pac_noise:
            gaussian_noise_covariance_path = f"{path_to_privatar}/experiment_scripts/pac_analysis/noise_covariance/pac_noise_outsource_ibdct_decoder_{num_freq_comp_outsourced}_{v}.npy"
        else: 
            gaussian_noise_covariance_path = f"{path_to_privatar}/experiment_scripts/dp_analysis/generated_dp_noise/dp_noise_ibdct_decoder_{num_freq_comp_outsourced}_{v}.npy"
        
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        print(f'python test_sgl_run_GH200.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_test {data_dir}/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name {wandb_author_name} --val_batch_size {val_batch_size} --num_freq_comp_outsourced {num_freq_comp_outsourced}   --gaussian_noise_covariance_path {gaussian_noise_covariance_path}')
        os.system(f'python test_sgl_run_GH200.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_test {data_dir}/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}   --arch base  --project_name {project_name} --author_name {wandb_author_name} --val_batch_size {val_batch_size} --num_freq_comp_outsourced {num_freq_comp_outsourced}   --gaussian_noise_covariance_path {gaussian_noise_covariance_path}')
