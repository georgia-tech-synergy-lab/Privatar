import os

# Please specify following configurations
#######
path_to_privatar = "/workspace/uwing2/Privatar"
wandb_author_name = "jimmytong"
sparsity_list = [0.2]#, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

data_dir = f"{path_to_privatar}/dataset/m--20180227--0000--6795937--GHS" 
initial_model_weights_path = f"{path_to_privatar}/pretrain_model/6795937_best_model.pth"
train_batch_size = 10
val_batch_size = 10
epochs = 2 
#######

# On syenrgy3 machine --- the following codes should be executed
# scl enable devtoolset-11 bash
#  --master_port=25678
for sparsity in sparsity_list:
    project_name = f'chnl_sparsity_0_{str(sparsity).split(".")[-1]}'
    result_path = f"{path_to_privatar}/training_results/{project_name}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    print(f'python3 train_GH200.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_train {data_dir}/frame_list.txt --framelist_test {data_dir}/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1  --arch base --unified_pruning_ratio {sparsity} --model_ckpt {initial_model_weights_path} --project_name {project_name} --author_name {wandb_author_name}')
    os.system(f'python3 train_GH200.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_train {data_dir}/frame_list.txt --framelist_test {data_dir}/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1  --arch base --unified_pruning_ratio {sparsity} --model_ckpt {initial_model_weights_path} --project_name {project_name} --author_name {wandb_author_name}')
