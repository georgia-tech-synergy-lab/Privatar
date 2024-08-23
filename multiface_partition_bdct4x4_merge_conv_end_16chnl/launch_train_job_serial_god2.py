import os

# Please select one path out of the following three paths based on the server
path_prefix = "/home/jianming/work/multiface/"
result_path_prefix = "/home/jianming/work/Privatar_prj/training_results/"
average_texture_path = "dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"
path_variance_matrix_tensor = "/tmp/dummy_path"
prefix_path_captured_latent_code = "/tmp/dummy_path"
pretrain_model_path = "/home/jianming/work/Privatar_prj/training_results/bdct4x4_hp_merge_pixel_concur_16_chnl_uw_train/best_model.pth"

average_texture_path = path_prefix + average_texture_path

# On syenrgy3 machine --- the following codes should be executed
result_path = f"{result_path_prefix}bdct4x4_hp_merge_pixel_concur_16_chnl"
# result_path = f"{result_path_prefix}bdct4x4_hp_merge_pixel_{str(BDCT_thres)}"
if not os.path.exists(result_path):
    os.makedirs(result_path)
print(f'python train.py --data_dir /scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS --krt_dir /scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train /scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test /scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/frame_list.txt --average_texture_path /scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --arch base  --project_name bdct_concur_all_16_chnls --author_name jimmytong  --path_variance_matrix_tensor {path_variance_matrix_tensor} --prefix_path_captured_latent_code {prefix_path_captured_latent_code} --model_ckpt {pretrain_model_path}')
os.system(f'python train.py --data_dir /scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS --krt_dir /scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train /scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test /scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/frame_list.txt --average_texture_path /scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1  --arch base  --project_name bdct_concur_all_16_chnls --author_name jimmytong --path_variance_matrix_tensor {path_variance_matrix_tensor} --prefix_path_captured_latent_code {prefix_path_captured_latent_code} --model_ckpt {pretrain_model_path}')
