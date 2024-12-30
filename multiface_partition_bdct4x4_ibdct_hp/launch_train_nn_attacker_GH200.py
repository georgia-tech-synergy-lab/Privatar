import os

# Please select one path out of the following three paths based on the server
data_dir = "/workspace/uwing2/multiface/dataset"
result_path_prefix = "/workspace/uwing2/Privatar/training_results/"

num_freq_comp_outsourced = 14
project_name = f"NN_attacker_{num_freq_comp_outsourced}"
best_model_path = f"/workspace/uwing2/Privatar/training_results/bdct_hp_ibdct_decoder_{num_freq_comp_outsourced}/best_model.pth"
framelist_test = "/workspace/uwing2/Privatar/experiment_scripts/empirical_attack/selected_expression_frame_list.txt"
camera_configs_path = "/workspace/uwing2/Privatar/experiment_scripts/empirical_attack/attack-camera-split-config_6795937.json"
epoch = 100
input_feature = 256
hidden_feature = 128
output_feature = 65
lr = 1e-4
momentum  = 0

noisy_attack = False
if noisy_attack:
    mi_list = [1, 0.1, 0.01]
    for v in mi_list:
        gaussian_noise_covariance_path = f"/workspace/uwing2/Privatar/experiment_scripts/pac_analysis/noise_covariance/noise_sigma_ibdct_decoder_{num_freq_comp_outsourced}_{v}.npy"

        result_path = f"{result_path_prefix}{project_name}_{lr}_{momentum}"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        print(f'python train_NN_attacker.py --data_dir {data_dir}/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/m--20180227--0000--6795937--GHS/KRT --framelist_test {framelist_test}  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --gaussian_noise_covariance_path {gaussian_noise_covariance_path} --camera_configs_path {camera_configs_path} --epoch {epoch} --input_feature {input_feature} --hidden_feature {hidden_feature} --output_feature {output_feature}  --lr {lr} --momentum {momentum}')
        os.system(f'python train_NN_attacker.py --data_dir {data_dir}/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/m--20180227--0000--6795937--GHS/KRT --framelist_test {framelist_test}  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong  --num_freq_comp_outsourced {num_freq_comp_outsourced} --gaussian_noise_covariance_path {gaussian_noise_covariance_path} --camera_configs_path {camera_configs_path} --epoch {epoch} --input_feature {input_feature} --hidden_feature {hidden_feature} --output_feature {output_feature} --lr {lr} --momentum {momentum}')
else:
    result_path = f"{result_path_prefix}{project_name}_{lr}_{momentum}"

    print(f'python train_NN_attacker.py --data_dir {data_dir}/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/m--20180227--0000--6795937--GHS/KRT --framelist_test {framelist_test}  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --camera_configs_path {camera_configs_path} --epoch {epoch} --input_feature {input_feature} --hidden_feature {hidden_feature} --output_feature {output_feature}  --lr {lr} --momentum {momentum}')
    os.system(f'python train_NN_attacker.py --data_dir {data_dir}/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/m--20180227--0000--6795937--GHS/KRT --framelist_test {framelist_test}  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong  --num_freq_comp_outsourced {num_freq_comp_outsourced} --camera_configs_path {camera_configs_path} --epoch {epoch} --input_feature {input_feature} --hidden_feature {hidden_feature} --output_feature {output_feature}  --lr {lr} --momentum {momentum}')
