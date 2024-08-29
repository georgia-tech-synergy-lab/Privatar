import os

# Please select one path out of the following three paths based on the server
data_dir = "/workspace/uwing2/multiface"
result_path_prefix = "/workspace/uwing2/Privatar/testing_results/"

val_batch_size = 24
bitwidth = 8

project_name = f"test_quantize_{bitwidth}"
result_path = f"{result_path_prefix}{project_name}"
best_model_path = f"/workspace/uwing2/Privatar/training_results/quant_{bitwidth}/best_model.pth"

if not os.path.exists(result_path):
    os.makedirs(result_path)
print(f'python test_sgl_run_uwing2.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --bitwidth {bitwidth} --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size}')
os.system(f'python test_sgl_run_uwing2.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --bitwidth {bitwidth} --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size}')
