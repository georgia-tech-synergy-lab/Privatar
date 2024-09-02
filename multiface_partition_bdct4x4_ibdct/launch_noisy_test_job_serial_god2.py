import os

# Please select one path out of the following three paths based on the server
data_dir = "/scratch2/multiface/dataset"
result_path_prefix = "/home/jianming/work/Privatar_prj/testing_results/"

val_batch_size = 20
project_name = f"noisy_test_bdct_hp_ibdct_decoder_0"
best_model_path = "/home/jianming/work/Privatar_prj/training_results/bdct_hp_ibdct_decoder_0/best_model.pth"
result_path = f"{result_path_prefix}{project_name}"

mi_list = [0.01, 0.1, 1]
for v in mi_list:
    gaussian_noise_covariance_path = f"/home/jianming/work/Privatar_prj/custom_scripts/pac_analysis/noise_covariance/noise_sigma_ibdct_decoder_0_{v}.npy"

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print(f'python test_sgl_run.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --gaussian_noise_covariance_path {gaussian_noise_covariance_path}')
    os.system(f'python test_sgl_run.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --gaussian_noise_covariance_path {gaussian_noise_covariance_path}')
