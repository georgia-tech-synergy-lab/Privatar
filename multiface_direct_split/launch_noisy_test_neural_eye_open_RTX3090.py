import os

# Please select one path out of the following three paths based on the server
data_dir = "/scratch2/multiface/dataset"
result_path_prefix = "/scratch2/jianming/work/Privatar_prj/testing_results/"

val_batch_size = 1
project_name = f"test_render_ds_noisy_sgl_expression_neural_eye_open"
best_model_path = "/scratch2/jianming/work/Privatar_prj/training_results/test_vae_direct_split/best_model.pth"
result_path = f"{result_path_prefix}{project_name}"
img_path = f"/scratch2/multiface/dataset/dataset/m--20180227--0000--6795937--GHS/images/E001_Neutral_Eyes_Open/400009/000102.png"

gaussian_noise_covariance_path = f"/scratch2/jianming/work/Privatar_prj/experiment_scripts/dp_analysis/generated_dp_noise/dp_noise_completed_outsourced_ibdct_decoder_1.npy"

if not os.path.exists(result_path):
    os.makedirs(result_path)
print(f'python test_sgl_expression_render.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --save_img --image_path {img_path} --gaussian_noise_covariance_path {gaussian_noise_covariance_path}')
os.system(f'python test_sgl_expression_render.py --data_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {data_dir}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --val_batch_size {val_batch_size} --save_img --image_path {img_path} --gaussian_noise_covariance_path {gaussian_noise_covariance_path}')
