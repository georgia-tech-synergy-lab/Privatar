import os

# Please select one path out of the following three paths based on the server
path_prefix_uwing2 = "/workspace/uwing2/multiface/"
# Please select one path out of the following three paths based on the server

#######
path_prefix = path_prefix_uwing2
model_ckpt = "/workspace/uwing2/multiface/pretrained_model/6795937_best_base_model.pth"
#######

train_batch_size = 10
val_batch_size = 10
num_freq_comp_outsourced_list = [14] #[2, 4, 6, 8, 10, 12, 14]
epochs = 2 

for num_freq_comp_outsourced in num_freq_comp_outsourced_list:
    project_name = f"bdct4x4_hp_ibdct_decode_{num_freq_comp_outsourced}"
    result_path = f"/workspace/uwing2/Privatar/training_results/bdct_hp_ibdct_decoder_{num_freq_comp_outsourced}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    print(f'python3 train_uwing2.py --data_dir {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1  --arch base --project_name {project_name} --author_name jimmytong --train_batch_size {train_batch_size} --val_batch_size {val_batch_size} --model_ckpt {model_ckpt} --num_freq_comp_outsourced {num_freq_comp_outsourced} --epochs {epochs}')
    os.system(f'python3 train_uwing2.py --data_dir {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1  --arch base --project_name {project_name} --author_name jimmytong --train_batch_size {train_batch_size} --val_batch_size {val_batch_size} --model_ckpt {model_ckpt} --num_freq_comp_outsourced {num_freq_comp_outsourced} --epochs {epochs}')
