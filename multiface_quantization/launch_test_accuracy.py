import os

# Please select one path out of the following three paths based on the server
data_dir = "/scratch2/multiface/dataset"
result_path_prefix = "/scratch2/jianming/work/Privatar_prj/render_results/"

val_batch_size = 32
bitwidth = 8 # could be 8 or 16, as torch only supports int8 and int16
project_name = f"test_accuracy_quant_{bitwidth}"
best_model_path = "/scratch2/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth"
result_path = f"{result_path_prefix}{project_name}"

if not os.path.exists(result_path):
    os.makedirs(result_path)
print(f'python test_accuracy.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --bitwidth {bitwidth}')
os.system(f'python test_accuracy.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --bitwidth {bitwidth}')
