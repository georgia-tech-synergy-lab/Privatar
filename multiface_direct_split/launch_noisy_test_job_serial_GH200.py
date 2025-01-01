import os

# !!! Note: this script requires training to be done (so that it could use the post-training model weights)
# Please select one path out of the following three paths based on the server

# Please specify following configurations
#######
using_pac_noise = False # True for using PAC noise, False for using DP noise
path_to_privatar = "/workspace/uwing2/Privatar"
project_name = f"test_multiface_direct_split"
wandb_author_name = "jimmytong"

data_dir = f"{path_to_privatar}/dataset/m--20180227--0000--6795937--GHS" 
result_path = f"{path_to_privatar}/testing_results/{project_name}"
post_training_model_path = f"{path_to_privatar}/training_results/multiface_direct_split/best_model.pth" 
val_batch_size = 10
#######

mi_list = [1, 0.1, 0.01]
for v in mi_list:
    if using_pac_noise:
        gaussian_noise_covariance_path = f"{path_to_privatar}/experiment_scripts/pac_analysis/noise_covariance/pac_noise_outsource_ibdct_decoder_16_{v}.npy"
    else: 
        gaussian_noise_covariance_path = f"{path_to_privatar}/experiment_scripts/dp_analysis/generated_dp_noise/dp_noise_ibdct_decoder_16_{v}.npy"
        
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print(f'python test_sgl_run_GH200.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_test {data_dir}/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {post_training_model_path}  --arch base  --project_name {project_name} --author_name {wandb_author_name} --val_batch_size {val_batch_size} --gaussian_noise_covariance_path {gaussian_noise_covariance_path}')
    os.system(f'python test_sgl_run_GH200.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_test {data_dir}/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {post_training_model_path}   --arch base  --project_name {project_name} --author_name {wandb_author_name} --val_batch_size {val_batch_size} --gaussian_noise_covariance_path {gaussian_noise_covariance_path}')
