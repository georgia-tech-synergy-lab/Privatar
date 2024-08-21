import os

# Please select one path out of the following three paths based on the server
path_prefix = "/home/jianming/work/multiface/"
dataset_prefix = "/scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS"
average_texture_path = "dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"
path_variance_matrix_tensor = "/home/jianming/work/Privatar_prj/testing_results/dummy_path" # Only useful for noisy training
prefix_path_captured_latent_code = "/home/jianming/work/Privatar_prj/testing_results/bdct4x4_hp_"
model_path = "/home/jianming/work/Privatar_prj/training_results/bdct4x4_horizontal_partition_0.3/best_model.pth"
#######

BDCT_threshold = [0.3] 
# BDCT_threshold = [0.4, 0.8, 1, 1.6, 2.4, 3, 4, 5, 6, 19] 

# On syenrgy3 machine --- the following codes should be executed
# scl enable devtoolset-11 bash
#  --master_port=25678
for BDCT_thres in BDCT_threshold:
    result_path = f"{prefix_path_captured_latent_code}{str(BDCT_thres)}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print(f'torchrun --nproc_per_node=1 --master_port=25678  test.py --data_dir {dataset_prefix} --krt_dir {dataset_prefix}/KRT --framelist_test {dataset_prefix}/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {model_path} --frequency_threshold {BDCT_thres} --project_name bdct4x4_hp_testing --author_name jimmytong --arch base --average_texture_path {path_prefix}{average_texture_path} --prefix_path_captured_latent_code {prefix_path_captured_latent_code} --save_latent_code_to_external_device')
    os.system(f'torchrun --nproc_per_node=1 --master_port=25678  test.py --data_dir {dataset_prefix} --krt_dir {dataset_prefix}/KRT --framelist_test {dataset_prefix}/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {model_path} --frequency_threshold {BDCT_thres} --project_name bdct4x4_hp_testing --author_name jimmytong --arch base --average_texture_path {path_prefix}{average_texture_path} --prefix_path_captured_latent_code {prefix_path_captured_latent_code} --save_latent_code_to_external_device')


