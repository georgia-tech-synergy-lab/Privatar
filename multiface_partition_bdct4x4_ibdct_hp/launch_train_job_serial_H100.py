import os

# Please select one path out of the following three paths based on the server
path_prefix_ice = "/storage/ice1/3/0/jtong45/multiface/"

average_texture_path = "dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"
# Please select one path out of the following three paths based on the server

#######
path_prefix = path_prefix_ice
#######

average_texture_path = path_prefix + average_texture_path

# On syenrgy3 machine --- the following codes should be executed
# scl enable devtoolset-11 bash
#  --master_port=25678
train_batch_size = 30
val_batch_size = 30
num_freq_comp_outsourced = 6
epochs = 2

model_ckpt = f"/storage/ice1/3/0/jtong45/Privatar/training_results/bdct_hp_ibdct_decoder_{num_freq_comp_outsourced}_secondrun/best_model.pth"
result_path = f"/storage/ice1/3/0/jtong45/Privatar/training_results/bdct_hp_ibdct_decoder_{num_freq_comp_outsourced}"
if not os.path.exists(result_path):
    os.makedirs(result_path)

print(f'python3 train_uwing2.py --data_dir {path_prefix_ice}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix_ice}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train {path_prefix_ice}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test {path_prefix_ice}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1  --arch base --project_name bdct4x4_hp_nn_decode_{num_freq_comp_outsourced} --author_name jimmytong --train_batch_size {train_batch_size} --val_batch_size {val_batch_size} --model_ckpt {model_ckpt} --num_freq_comp_outsourced {num_freq_comp_outsourced}  --epochs {epochs} >> {result_path}/log')
os.system(f'python3 train_uwing2.py --data_dir {path_prefix_ice}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix_ice}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train {path_prefix_ice}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test {path_prefix_ice}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1  --arch base --project_name bdct4x4_hp_nn_decode_{num_freq_comp_outsourced} --author_name jimmytong --train_batch_size {train_batch_size} --val_batch_size {val_batch_size} --model_ckpt {model_ckpt} --num_freq_comp_outsourced {num_freq_comp_outsourced}  --epochs {epochs} >> {result_path}/log')
