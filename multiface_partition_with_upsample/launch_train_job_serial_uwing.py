import os

# Please select one path out of the following three paths based on the server
path_prefix_uwing2 = "/workspace/uwing2/multiface/"

average_texture_path = "dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"
# Please select one path out of the following three paths based on the server

#######
path_prefix = path_prefix_uwing2
#######

average_texture_path = path_prefix + average_texture_path

enc_dec_list = [["dec"]]
BDCT_threshold = [0.4, 0.8, 1]#, 1, 1.6, 2.4, 3, 4, 5, 6, 19] 
# BDCT_threshold = [0.4, 0.8, 1, 1.6, 2.4, 3, 4, 5, 6, 19] 
# BDCT_threshold = BDCT_threshold[::-1]
# BDCT_threshold = BDCT_threshold[2:] # 5, 4, 3 ..

# On syenrgy3 machine --- the following codes should be executed
# scl enable devtoolset-11 bash
#  --master_port=25678
for BDCT_thres in BDCT_threshold:
    result_path = f"/workspace/uwing2/Privatar/training_results/horizontal_partition_{str(BDCT_thres)}"
    pretrained_model_path = f"/workspace/uwing2/Privatar/training_results/horizontal_partition_{str(BDCT_thres)}/best_model.pth"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    print(f'python3 train_uwing2.py --data_dir {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --average_texture_path {average_texture_path} --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt {pretrained_model_path} --arch base --frequency_threshold {BDCT_thres} --project_name hp_training --author_name jimmytong --apply_gaussian_noise False --save_latent_code_to_external_device False')
    os.system(f'python3 train_uwing2.py --data_dir {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test {path_prefix_uwing2}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --average_texture_path {average_texture_path} --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt {pretrained_model_path} --arch base --frequency_threshold {BDCT_thres} --project_name hp_training --author_name jimmytong --apply_gaussian_noise False --save_latent_code_to_external_device False')
