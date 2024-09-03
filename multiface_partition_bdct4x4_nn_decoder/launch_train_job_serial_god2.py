import os

# Please select one path out of the following three paths based on the server
path_prefix = "/home/jianming/work/multiface/"
result_path_prefix = "/home/jianming/work/Privatar_prj/training_results/"
average_texture_path = "dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"
path_variance_matrix_tensor = "/tmp/dummy_path"
prefix_path_captured_latent_code = "/tmp/dummy_path"
model_ckpt = "/home/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth"

average_texture_path = path_prefix + average_texture_path
train_batch_size = 10
val_batch_size = 10
num_freq_comp_outsourced = 0
epochs = 2

# On syenrgy3 machine --- the following codes should be executed
result_path = f"/home/jianming/work/Privatar_prj/training_results/bdct_hp_nn_decoder_{num_freq_comp_outsourced}"
if not os.path.exists(result_path):
    os.makedirs(result_path)
print(f'python train.py --data_dir /scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS --krt_dir /scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train /scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test /scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/frame_list.txt --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --arch base  --project_name bdct_hp_nn_decoder_0 --author_name jimmytong  --model_ckpt {model_ckpt} --train_batch_size {train_batch_size} --val_batch_size {val_batch_size} --epochs {epochs}')
os.system(f'python train.py --data_dir /scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS --krt_dir /scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train /scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test /scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/frame_list.txt --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1  --arch base  --project_name bdct_hp_nn_decoder_0 --author_name jimmytong  --model_ckpt {model_ckpt} --train_batch_size {train_batch_size} --val_batch_size {val_batch_size} --epochs {epochs}')
