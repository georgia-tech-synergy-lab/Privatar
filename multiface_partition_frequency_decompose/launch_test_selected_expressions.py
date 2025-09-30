import os

# Please select one path out of the following three paths based on the server
data_dir = "/work"
result_path_prefix = "/work/render_results/"

val_batch_size = 1
num_freq_comp_offloaded_list = [2, 4, 6, 8, 10, 12, 14]
for num_freq_comp_offloaded in num_freq_comp_offloaded_list:
  project_name = f"partition_{num_freq_comp_offloaded}"
  best_model_path = f"/work/training_results/partition_{num_freq_comp_offloaded}/best_model.pth"
  result_path = f"{result_path_prefix}{project_name}"
  selected_frame_path = "/work/experiment_scripts/render_scripts/selected_frame.txt"
  selected_camera = "400023" # the direct view camera id

  if not os.path.exists(result_path):
      os.makedirs(result_path)
  print(f'python test_selected_expressions.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {selected_frame_path} --result_path {result_path} --test_segment "/work/experiment_scripts/test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --save_img --num_freq_comp_offloaded {num_freq_comp_offloaded} --selected_camera {selected_camera}')
  os.system(f'python test_selected_expressions.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {selected_frame_path} --result_path {result_path} --test_segment "/work/experiment_scripts/test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --save_img --num_freq_comp_offloaded {num_freq_comp_offloaded} --selected_camera {selected_camera}')
