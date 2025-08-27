import os

# Please select one path out of the following three paths based on the server
data_dir = "/scratch2/multiface/dataset"
result_path_prefix = "/scratch2/jianming/work/Privatar_prj/testing_results/"

val_batch_size = 32
project_name = f"test_accuracy_HP_all_expressions_co"
best_model_path = f"/scratch2/jianming/work/Privatar_prj/training_results/test_vae_direct_split/best_model.pth"
noise_covariance_path = f"/scratch2/jianming/work/Privatar_prj/experiment_scripts/dp_analysis/generated_dp_noise/dp_noise_completed_outsourced_ibdct_decoder_1.npy"
result_path = f"{result_path_prefix}{project_name}"
if not os.path.exists(result_path):
    os.makedirs(result_path)
print(f'python test_accuracy.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --gaussian_noise_covariance_path {noise_covariance_path}')
os.system(f'python test_accuracy.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --gaussian_noise_covariance_path {noise_covariance_path}')
