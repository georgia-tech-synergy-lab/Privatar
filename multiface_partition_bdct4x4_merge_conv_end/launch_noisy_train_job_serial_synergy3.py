import os

# Please select one path out of the following three paths based on the server
path_prefix = "/usr/scratch/jianming/multiface/"
pretrain_model_path_prefix = "/usr/scratch/jianming/Privatar/training_results/horizontal_partition_"
result_path_prefix = "/usr/scratch/jianming/Privatar/training_results"
prefix_path_captured_latent_code = "/usr/scratch/jianming/Privatar/testing_results/horizontal_partition_"
average_texture_path = "dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"
# Please select one path out of the following three paths based on the server

average_texture_path = path_prefix + average_texture_path

BDCT_threshold = [3.5]
mi_list = [1]

# On syenrgy3 machine --- the following codes should be executed
# scl enable devtoolset-11 bash
#  --master_port=25678
for mi_val in mi_list:
    for BDCT_thres in BDCT_threshold:
        result_path = f"{result_path_prefix}/noisy_bdct4x4_hp_{str(BDCT_thres)}_{str(mi_val)}"
        path_variance_matrix_tensor = f"/usr/scratch/jianming/Privatar/profiled_latent_code/statistics/bdct_4x4_noisy_hp_{BDCT_thres}_mutual_bound_{mi_val}.pth"
        best_model_path = f"/usr/scratch/jianming/Privatar/training_results/bdct4x4_horizontal_partition_{str(BDCT_thres)}/best_model.pth"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        print(f'CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=25681 train.py --data_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --average_texture_path {path_prefix}dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt {best_model_path} --arch base --frequency_threshold {BDCT_thres} --project_name noisy_bdct4x4_hp --author_name jimmytong --prefix_path_captured_latent_code {prefix_path_captured_latent_code} --apply_gaussian_noise --path_variance_matrix_tensor {path_variance_matrix_tensor}')
        os.system(f'CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=25681 train.py --data_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --average_texture_path {path_prefix}dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt {best_model_path} --arch base --frequency_threshold {str(BDCT_thres)} --project_name noisy_bdct4x4_hp --author_name jimmytong --prefix_path_captured_latent_code {prefix_path_captured_latent_code} --apply_gaussian_noise --path_variance_matrix_tensor {path_variance_matrix_tensor}')
