import os

# Please select one path out of the following three paths based on the server
data_dir = "/scratch2/multiface/dataset"
result_path_prefix = "/home/jianming/work/Privatar_prj/testing_results/"

val_batch_size = 16
unified_pruning_ratio = 0.1

project_name = f"test_sparse_{unified_pruning_ratio}"
result_path = f"{result_path_prefix}{project_name}"
best_model_path = f"/home/jianming/work/Privatar_prj/training_results/sparse_{str(unified_pruning_ratio).replace('.', '_')}/best_model.pth"

if not os.path.exists(result_path):
    os.makedirs(result_path)
print(f'python test_sgl_run.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size}')
os.system(f'python test_sgl_run.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size}')
