import os

# Please specify following configurations
#######
path_prefix_GH200 = "/workspace/uwing2/multiface/"
path_to_privatar = "/workspace/uwing2/Privatar"
project_name = f"multiface_direct_split"
wandb_author_name = "jimmytong"

data_dir = f"{path_to_privatar}/dataset/m--20180227--0000--6795937--GHS" 
result_path = f"{path_to_privatar}/training_results/{project_name}"
initial_model_weights_path = f"{path_to_privatar}/pretrain_model/6795937_best_model.pth"
train_batch_size = 10
val_batch_size = 10
epochs = 2 
#######

result_path = f"/workspace/uwing2/Privatar/training_results/{project_name}"
if not os.path.exists(result_path):
    os.makedirs(result_path)

print(f'python3 train_GH200.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_train {data_dir}/frame_list.txt --framelist_test {data_dir}/frame_list.txt --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1  --arch base --project_name {project_name} --author_name {wandb_author_name} --train_batch_size {train_batch_size} --val_batch_size {val_batch_size} --model_ckpt {initial_model_weights_path} --epochs {epochs}')
os.system(f'python3 train_GH200.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_train {data_dir}/frame_list.txt --framelist_test {data_dir}/frame_list.txt --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1  --arch base --project_name {project_name} --author_name {wandb_author_name} --train_batch_size {train_batch_size} --val_batch_size {val_batch_size} --model_ckpt {initial_model_weights_path} --epochs {epochs}')
