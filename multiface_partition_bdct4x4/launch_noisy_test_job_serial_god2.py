import os

# Please select one path out of the following three paths based on the server
path_prefix ="/home/jianming/work/multiface/"
data_prefix = "/scratch2/multiface/dataset"
path_testing_latent_code = "/home/jianming/work/Privatar_prj/testing_results/dummy_path"
average_texture_path = "dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"
path_variance_matrix_tensor = "/home/jianming/work/Privatar_prj/profiled_latent_code/statistics/bdct4x4_hp_noise_variance_matrix_0.3.pth"

BDCT_threshold = [0.3] 

# On syenrgy3 machine --- the following codes should be executed
# scl enable devtoolset-11 bash
#  --master_port=25678
for BDCT_thres in BDCT_threshold:
    result_path = f"{path_testing_latent_code}{str(BDCT_thres)}"
    best_model_path = f"/home/jianming/work/Privatar_prj/training_results/noisy_bdct4x4_hp_{BDCT_thres}/best_model.pth"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print(f'torchrun --nproc_per_node=1 --master_port=25678  test.py --data_dir {data_prefix}/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {path_prefix}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path} --frequency_threshold {BDCT_thres} --project_name noisy_bdct_4x4_hp_ --author_name jimmytong --arch base --average_texture_path {path_prefix}{average_texture_path} --prefix_path_captured_latent_code {path_testing_latent_code} --apply_gaussian_noise --path_variance_matrix_tensor {path_variance_matrix_tensor}')
    os.system(f'torchrun --nproc_per_node=1 --master_port=25678  test.py --data_dir {data_prefix}/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {path_prefix}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path} --frequency_threshold {BDCT_thres}  --project_name noisy_bdct_4x4_hp_ --author_name jimmytong --arch base --average_texture_path {path_prefix}{average_texture_path} --prefix_path_captured_latent_code {path_testing_latent_code} --apply_gaussian_noise --path_variance_matrix_tensor {path_variance_matrix_tensor}')
