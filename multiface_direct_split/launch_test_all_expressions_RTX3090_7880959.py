import os

# Please select one path out of the following three paths based on the server
data_dir = "/scratch2/multiface/dataset"
result_path_prefix = "/scratch2/jianming/work/Privatar_prj/render_results/"

val_batch_size = 1
project_name = f"original_vae_direct_spli_7880959"
best_model_path = "/scratch2/jianming/work/Privatar_prj/training_results/multiface_direct_split_7889059/best_model.pth"
result_path = f"{result_path_prefix}{project_name}"
img_paths = "/scratch2/jianming/work/Privatar_prj/experiment_scripts/render_scripts/test_image_path"

if not os.path.exists(result_path):
    os.makedirs(result_path)
print(f'python test_many_expression_render.py --data_dir {data_dir}/dataset/m--20180927--0000--7889059--GHS --krt_dir {data_dir}/dataset/m--20180927--0000--7889059--GHS/KRT --framelist_test {data_dir}/dataset/m--20180927--0000--7889059--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --save_latent_code --save_img --image_paths {img_paths}')
os.system(f'python test_many_expression_render.py --data_dir {data_dir}/dataset/m--20180927--0000--7889059--GHS --krt_dir {data_dir}/dataset/m--20180927--0000--7889059--GHS/KRT --framelist_test {data_dir}/dataset/m--20180927--0000--7889059--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --save_latent_code --save_img --image_paths {img_paths}')
