import os

# Please select one path out of the following three paths based on the server
average_texture_path = "dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"

#######
path_prefix = "/storage/ice1/3/0/jtong45/multiface/"
path_variance_matrix_tensor = "/tmp/dummy_path"
prefix_path_captured_latent_code = "/tmp/dummy"
best_model_path = "/storage/ice1/3/0/jtong45/Privatar/training_results/bdct4x4_hp_merge_conv_16chnl_first_train/best_model.pth"
# model_ckpt = "/workspace/uwing2/Privatar/training_results/bdct_4x4_hp_0.35_first_train/best_model.pth"
#######

average_texture_path = path_prefix + average_texture_path

result_path = f"/storage/ice1/3/0/jtong45/Privatar/training_results/bdct4x4_hp_merge_conv_16chnl"
if not os.path.exists(result_path):
    os.makedirs(result_path)

print(f'python3 train_uwing2.py --data_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --average_texture_path {average_texture_path} --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1  --arch base  --project_name bdct_concur_all_16_chnls --author_name jimmytong --path_variance_matrix_tensor {path_variance_matrix_tensor} --prefix_path_captured_latent_code {path_variance_matrix_tensor} --model_ckpt {best_model_path}')
os.system(f'python3 train_uwing2.py --data_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --average_texture_path {average_texture_path} --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1  --arch base --project_name bdct_concur_all_16_chnls --author_name jimmytong --path_variance_matrix_tensor {path_variance_matrix_tensor} --prefix_path_captured_latent_code {path_variance_matrix_tensor} --model_ckpt {best_model_path}')
