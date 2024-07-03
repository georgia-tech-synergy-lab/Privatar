import os

# Please select one path out of the following three paths based on the server
path_prefix_uwing2 = "/workspace/uwing2/multiface/"

average_texture_path = "dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"
# Please select one path out of the following three paths based on the server

#######
path_prefix = path_prefix_uwing2
path_variance_matrix_tensor = "/workspace/uwing2/Privatar/testing_results/dummy_path"
prefix_path_captured_latent_code = ""
model_ckpt = "/workspace/uwing2/Privatar/training_results/bdct_4x4_hp_0.35_first_train/best_model.pth"
#######

average_texture_path = path_prefix + average_texture_path
BDCT_threshold = [0.1, 0.3, 0.305, 0.35, 0.4, 0.42, 0.45, 0.5, 0.6, 0.63, 0.7, 1.035, 1.1, 1.2, 3.5, 5]

# On syenrgy3 machine --- the following codes should be executed
# scl enable devtoolset-11 bash
#  --master_port=25678
for BDCT_thres in BDCT_threshold:
    result_path = f"/workspace/uwing2/Privatar/training_results/bdct_4x4_nohp_{str(BDCT_thres)}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    print(f'python3 train_uwing2.py --data_dir {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --average_texture_path {average_texture_path} --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1  --arch base --frequency_threshold {BDCT_thres} --project_name bdct4x4_hp_training --author_name jimmytong --path_variance_matrix_tensor {path_variance_matrix_tensor} --prefix_path_captured_latent_code {path_variance_matrix_tensor} --model_ckpt {model_ckpt}')
    os.system(f'python3 train_uwing2.py --data_dir {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --average_texture_path {average_texture_path} --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1  --arch base --frequency_threshold {BDCT_thres} --project_name bdct4x4_hp_training --author_name jimmytong --path_variance_matrix_tensor {path_variance_matrix_tensor} --prefix_path_captured_latent_code {path_variance_matrix_tensor} --model_ckpt {model_ckpt}')
