import os

# Please select one path out of the following three paths based on the server
data_dir = "/scratch2/multiface/dataset"
result_path_prefix = "/scratch2/jianming/work/Privatar_prj/testing_results/"

val_batch_size = 1
for bitwidth in [2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
  project_name = f"test_quantize_neural_eye_open_{bitwidth}"
  best_model_path = f"/scratch2/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth"
  # best_model_path ="/scratch2/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth"
  result_path = f"/scratch2/jianming/work/Privatar_prj/testing_results/{project_name}"
  img_path = f"/scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/images/E001_Neutral_Eyes_Open/400009/000102.png"

  print(f'python test_sgl_expression_render.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --save_latent_code --image_path {img_path} --bitwidth {bitwidth} --save_img --result_path {result_path}')
  os.system(f'python test_sgl_expression_render.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --save_latent_code --image_path {img_path} --bitwidth {bitwidth} --save_img --result_path {result_path}')
