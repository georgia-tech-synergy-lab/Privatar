import os

# Please select one path out of the following three paths based on the server
data_dir = "/workspace/uwing2/multiface"
result_path_prefix = "/workspace/uwing2/Privatar/testing_results/"


# Please specify following configurations
#######
path_to_privatar = "/workspace/uwing2/Privatar"
project_name = f"test_bdct_hp_ibdct_decoder_0"
wandb_author_name = "jimmytong"

data_dir = f"{path_to_privatar}/dataset/m--20180227--0000--6795937--GHS" 
result_path = f"{path_to_privatar}/testing_results/{project_name}"
post_training_model_path = f"{path_to_privatar}/training_results/bdct_hp_ibdct_decoder_0/model.pth" 
val_batch_size = 10
#######

if not os.path.exists(result_path):
    os.makedirs(result_path)
print(f'python test_sgl_run_GH200.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_test {data_dir}/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {post_training_model_path}  --arch base  --project_name {project_name} --author_name {wandb_author_name} --val_batch_size {val_batch_size}  --save_latent_code')
os.system(f'python test_sgl_run_GH200.py --data_dir {data_dir} --krt_dir {data_dir}/KRT --framelist_test {data_dir}/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {post_training_model_path}   --arch base  --project_name {project_name} --author_name {wandb_author_name} --val_batch_size {val_batch_size}  --save_latent_code')
