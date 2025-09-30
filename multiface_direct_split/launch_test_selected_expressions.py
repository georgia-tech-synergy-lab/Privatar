import os

# Please select one path out of the following three paths based on the server
data_dir = "/work"
result_path_prefix = "/work/render_results/"

val_batch_size = 1
project_name = f"multiface_direct_split"
best_model_path = f"/work/training_results/multiface_direct_split/best_model.pth"
result_path = f"{result_path_prefix}{project_name}"
selected_frame_path = "/work/experiment_scripts/render_scripts/selected_frame.txt"
selected_camera = "400023" # the direct view camera id

if not os.path.exists(result_path):
    os.makedirs(result_path)
print(f'python test_selected_expressions.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {selected_frame_path} --result_path {result_path} --test_segment "/work/experiment_scripts/test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --save_img --selected_camera {selected_camera}')
os.system(f'python test_selected_expressions.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {selected_frame_path} --result_path {result_path} --test_segment "/work/experiment_scripts/test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --save_img --selected_camera {selected_camera}')
