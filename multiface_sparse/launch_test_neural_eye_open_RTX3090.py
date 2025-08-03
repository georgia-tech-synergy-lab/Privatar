import os

# Please select one path out of the following three paths based on the server
data_dir = "/scratch2/multiface/dataset"
result_path_prefix = "/scratch2/jianming/work/Privatar_prj/testing_results/"

val_batch_size = 1
for unified_pruning_ratio in [0.6, 0.7, 0.8]:
  project_name = f"test_sparse_neural_eye_open_{unified_pruning_ratio}"
  best_model_path = f"/scratch2/jianming/work/Privatar_prj/training_results/sparse_0_{unified_pruning_ratio*10%10:0.0f}/best_model.pth"
  # best_model_path ="/scratch2/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth"
  result_path = f"/scratch2/jianming/work/Privatar_prj/testing_results/{project_name}"
  img_path = f"/scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/images/E001_Neutral_Eyes_Open/400009/000102.png"

  print(f'python test_sgl_expression_render.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --save_latent_code --image_path {img_path} --unified_pruning_ratio {unified_pruning_ratio} --save_img --result_path {result_path}')
  os.system(f'python test_sgl_expression_render.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --save_latent_code --image_path {img_path} --unified_pruning_ratio {unified_pruning_ratio} --save_img --result_path {result_path}')
