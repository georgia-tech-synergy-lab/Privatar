import os

# Please specify following configurations
#######
path_prefix_GH200 = "/scratch2/multiface/"
path_to_privatar = "/scratch2/jianming/work/Privatar_prj"
project_name = f"multiface_direct_split_7889059"
wandb_author_name = "jimmytong"

data_dir = "/scratch2/multiface/dataset/m--20180927--0000--7889059--GHS" 
result_path = f"{path_to_privatar}/training_results/{project_name}"
initial_model_weights_path = "/scratch2/jianming/work/Privatar_prj/training_results/pretrain_7889059/best_model.pth"
train_batch_size = 10
val_batch_size = 10
epochs = 2 
#######

result_path = f"/scratch2/jianming/work/Privatar_prj/training_results/{project_name}"
if not os.path.exists(result_path):
    os.makedirs(result_path)

print(f'python3 train_GH200.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_train {data_dir}/frame_list.txt --framelist_test {data_dir}/frame_list.txt --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1  --arch base --project_name {project_name} --author_name {wandb_author_name} --train_batch_size {train_batch_size} --val_batch_size {val_batch_size} --model_ckpt {initial_model_weights_path} --epochs {epochs}')
os.system(f'python3 train_GH200.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_train {data_dir}/frame_list.txt --framelist_test {data_dir}/frame_list.txt --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1  --arch base --project_name {project_name} --author_name {wandb_author_name} --train_batch_size {train_batch_size} --val_batch_size {val_batch_size} --model_ckpt {initial_model_weights_path} --epochs {epochs}')
