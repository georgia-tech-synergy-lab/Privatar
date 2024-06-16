import os

# Please select one path out of the following three paths based on the server
path_prefix_god2 = "/home/jianming/work/multiface/"
path_prefix_synergy3 = "{path_prefix}"

average_texture_path = "dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"
# Please select one path out of the following three paths based on the server
#######
path_prefix = path_prefix_god2
#######

average_texture_path = path_prefix + average_texture_path
model_list = ["DeepAppearanceVAE_FullModel_Frequency_Division"]

enc_dec_list = [["dec"]]
BDCT_threshold = [0.4, 0.8, 1] 
# BDCT_threshold = [0.4, 0.8, 1, 1.6, 2.4, 3, 4, 5, 6, 19] 
# BDCT_threshold = BDCT_threshold[::-1]
# BDCT_threshold = BDCT_threshold[2:] # 5, 4, 3 ..

# On syenrgy3 machine --- the following codes should be executed
# scl enable devtoolset-11 bash
#  --master_port=25678
for BDCT_thres in BDCT_threshold:
    result_path = f"{path_prefix_synergy3}/training_results/horizontal_partition_{str(BDCT_thres)}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print(f'torchrun --nproc_per_node=1 --master_port=25679 train.py --data_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --average_texture_path {path_prefix}dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt "{path_prefix}pretrained_model/6795937_best_base_model.pth" --arch base --frequency_threshold {BDCT_thres} --project_name hp_training --author_name jimmytong --apply_gaussian_noise False --save_latent_code_to_external_device False')
    os.system(f'torchrun --nproc_per_node=1 --master_port=25679 train.py --data_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test {path_prefix}dataset/m--20180227--0000--6795937--GHS/frame_list.txt --average_texture_path {path_prefix}dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt "{path_prefix}pretrained_model/6795937_best_base_model.pth" --arch base --frequency_threshold {BDCT_thres} --project_name hp_training --author_name jimmytong ')
