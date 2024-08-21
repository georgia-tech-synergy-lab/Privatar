import os

# Please select one path out of the following three paths based on the server
path_prefix = "/home/jianming/work/multiface/"
dataset_prefix = "/scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS"
result_path_prefix = "/home/jianming/work/Privatar_prj/training_results"
prefix_path_captured_latent_code = "/home/jianming/work/Privatar_prj/testing_results/dummy_path"
average_texture_path = "dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"
# Please select one path out of the following three paths based on the server
average_texture_path = path_prefix + average_texture_path

BDCT_threshold = [0.3]
mi_list = [0.01, 0.1, 1]
# BDCT_threshold = [0.4, 0.8, 1, 1.6, 2.4, 3, 4, 5, 6, 19] 
# BDCT_threshold = BDCT_threshold[::-1]
# BDCT_threshold = BDCT_threshold[2:] # 5, 4, 3 ..

# On syenrgy3 machine --- the following codes should be executed
# scl enable devtoolset-11 bash
#  --master_port=25678
for mi_val in mi_list:
    for BDCT_thres in BDCT_threshold:
        pretrain_model_path = f"/home/jianming/work/Privatar_prj/training_results/bdct4x4_hp_{str(BDCT_thres)}/best_model.pth"
        path_variance_matrix_tensor = f"/home/jianming/work/Privatar_prj/profiled_latent_code/statistics/bdct_4x4_noisy_hp_{str(BDCT_thres)}_mutual_bound_{str(mi_val)}.pth"
        result_path = f"{result_path_prefix}/noisy_bdct4x4_hp_{str(BDCT_thres)}_{str(mi_val)}"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        print(f'torchrun --nproc_per_node=1 --master_port=25680 train.py --data_dir {dataset_prefix} --krt_dir {dataset_prefix}/KRT --framelist_train {dataset_prefix}/frame_list.txt --framelist_test {dataset_prefix}/frame_list.txt --average_texture_path {dataset_prefix}/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt "{pretrain_model_path}" --arch base --frequency_threshold {BDCT_thres} --project_name bdct_4x4_noisy_hp_training --author_name jimmytong --prefix_path_captured_latent_code {prefix_path_captured_latent_code} --apply_gaussian_noise --path_variance_matrix_tensor {path_variance_matrix_tensor} --epochs 1')
        os.system(f'torchrun --nproc_per_node=1 --master_port=25680 train.py --data_dir {dataset_prefix} --krt_dir {dataset_prefix}/KRT --framelist_train {dataset_prefix}/frame_list.txt --framelist_test {dataset_prefix}/frame_list.txt --average_texture_path {dataset_prefix}/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt "{pretrain_model_path}" --arch base --frequency_threshold {str(BDCT_thres)} --project_name bdct_4x4_noisy_hp_training --author_name jimmytong --prefix_path_captured_latent_code {prefix_path_captured_latent_code} --apply_gaussian_noise --path_variance_matrix_tensor {path_variance_matrix_tensor} --epochs 1')
