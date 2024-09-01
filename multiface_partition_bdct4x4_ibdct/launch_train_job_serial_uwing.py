import os

# Please select one path out of the following three paths based on the server
path_prefix_uwing2 = "/workspace/uwing2/multiface/"

average_texture_path = "dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"
# Please select one path out of the following three paths based on the server

#######
path_prefix = path_prefix_uwing2
path_variance_matrix_tensor = "/workspace/uwing2/Privatar/testing_results/dummy_path"
prefix_path_captured_latent_code = ""
model_ckpt = "/workspace/uwing2/multiface/pretrained_model/6795937_best_base_model.pth"
#######

average_texture_path = path_prefix + average_texture_path

train_batch_size = 2
val_batch_size = 2
epochs = 1

result_path = f"/workspace/uwing2/Privatar/training_results/bdct_nn_decoder"
if not os.path.exists(result_path):
    os.makedirs(result_path)

print(f'python3 train_uwing2.py --data_dir {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --average_texture_path {average_texture_path} --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1  --arch base --project_name bdct4x4_hp_training --author_name jimmytong --train_batch_size {train_batch_size} --val_batch_size {val_batch_size} --model_ckpt {model_ckpt} --epochs {epochs}')
os.system(f'python3 train_uwing2.py --data_dir {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --average_texture_path {average_texture_path} --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1  --arch base --project_name bdct4x4_hp_training --author_name jimmytong --train_batch_size {train_batch_size} --val_batch_size {val_batch_size} --model_ckpt {model_ckpt} --epochs {epochs}')
