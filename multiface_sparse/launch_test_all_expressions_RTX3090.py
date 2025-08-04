import os

# Please select one path out of the following three paths based on the server
data_dir = "/scratch2/multiface/dataset"
result_path_prefix = "/scratch2/jianming/work/Privatar_prj/render_results/"

val_batch_size = 1
unified_pruning_ratio = 0.1
project_name = f"sparse_{unified_pruning_ratio}"
best_model_path = f"/scratch2/jianming/work/Privatar_prj/training_results/sparse_0_{unified_pruning_ratio*10%10:0.0f}/best_model.pth"
result_path = f"{result_path_prefix}{project_name}"
img_paths = "/scratch2/jianming/work/Privatar_prj/experiment_scripts/render_scripts/test_image_path"

if not os.path.exists(result_path):
    os.makedirs(result_path)
print(f'python test_many_expression_render.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --save_latent_code --save_img --image_paths {img_paths} --unified_pruning_ratio {unified_pruning_ratio}')
os.system(f'python test_many_expression_render.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --save_latent_code --save_img --image_paths {img_paths} --unified_pruning_ratio {unified_pruning_ratio}')
