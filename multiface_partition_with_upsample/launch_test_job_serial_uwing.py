import os

# Please select one path out of the following three paths based on the server
path_prefix_uwing2 = "/workspace/uwing2/multiface/"

path_testing_latent_code_uwing2 = "/workspace/uwing2/Privatar/testing_results/horizontal_partition_"
average_texture_path = "dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"
pretrain_model_path_prefix = "/workspace/uwing2/Privatar/training_results/horizontal_partition_"


#######
path_prefix = path_prefix_uwing2
path_testing_latent_code = path_testing_latent_code_uwing2
#######

BDCT_threshold = [1] 
# BDCT_threshold = [0.4, 0.8, 1, 1.6, 2.4, 3, 4, 5, 6, 19] 

# On syenrgy3 machine --- the following codes should be executed
# scl enable devtoolset-11 bash
#  --master_port=25678
for BDCT_thres in BDCT_threshold:
    result_path = f"{path_testing_latent_code}{str(BDCT_thres)}"
    best_model_path = f"{pretrain_model_path_prefix}{BDCT_thres}/best_model.pth"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print(f'python3 test_uwing2.py --data_dir {path_prefix}/dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {path_prefix}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path} --frequency_threshold {BDCT_thres} --project_name pretrain_testing --author_name jimmytong --arch base --average_texture_path {path_prefix}{average_texture_path} --prefix_path_captured_latent_code {path_testing_latent_code} --noisy_training False --save_latent_code_to_external_device True')
    os.system(f'python3 test_uwing2.py --data_dir {path_prefix}/dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {path_prefix}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path} --frequency_threshold {BDCT_thres} --project_name pretrain_testing --author_name jimmytong --arch base --average_texture_path {path_prefix}{average_texture_path} --prefix_path_captured_latent_code {path_testing_latent_code} --noisy_training False --save_latent_code_to_external_device True')
