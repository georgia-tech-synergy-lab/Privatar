import os

# Please specify following configurations
#######
path_to_privatar = "/workspace/uwing2/Privatar"
wandb_author_name = "jimmytong"
bitwidth_list = [15] # [8,9,10,11,12,13,14,15,16]

data_dir = f"{path_to_privatar}/dataset/m--20180227--0000--6795937--GHS" 
val_batch_size = 10
#######

for bitwidth in bitwidth_list:
    project_name = f"quant_{bitwidth}" 
    result_path = f"{path_to_privatar}/testing_results/{project_name}"
    post_training_model_path = f"{path_to_privatar}/training_results/{project_name}/best_model.pth"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    print(f'python test_sgl_run_GH200.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_test {data_dir}/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {post_training_model_path}  --bitwidth {bitwidth} --arch base  --project_name {project_name} --author_name {wandb_author_name} --val_batch_size {val_batch_size}')
    os.system(f'python test_sgl_run_GH200.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_test {data_dir}/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {post_training_model_path}  --bitwidth {bitwidth} --arch base  --project_name {project_name} --author_name {wandb_author_name} --val_batch_size {val_batch_size}')
