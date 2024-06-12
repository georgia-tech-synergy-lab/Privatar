import os

# Please select one path out of the following three paths based on the server
path_prefix_god2 ="/home/jianming/work/multiface/"
path_prefix_synergy3 = "/home/jianming/work/multiface/"
path_prefix_uwing2 = "/workspace/uwing2/multiface/"

path_testing_latent_code_god2 = "/home/jianming/work/Privatar_prj/testing_results/horizontal_partition_"
path_testing_latent_code_synergy3 = "/usr/scratch/jianming/Privatar/testing_results/horizontal_partition_"
average_texture_path = "dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png"

#######
path_prefix = path_prefix_synergy3
path_testing_latent_code = path_testing_latent_code_synergy3
#######

BDCT_threshold = [2.4] 
# BDCT_threshold = [0.4, 0.8, 1, 1.6, 2.4, 3, 4, 5, 6, 19] 

# On syenrgy3 machine --- the following codes should be executed
# scl enable devtoolset-11 bash
#  --master_port=25678
for BDCT_thres in BDCT_threshold:
    result_path = f"{path_testing_latent_code_god2}{str(BDCT_thres)}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print(f'torchrun --nproc_per_node=1 --master_port=25678  test.py --data_dir {path_prefix}/dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {path_prefix}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path "/home/jianming/work/Privatar_prj/training_results/horizontal_partition_{BDCT_thres}/best_model.pth" --frequency_threshold {BDCT_thres} --arch base --average_texture_path {path_prefix}{average_texture_path} --prefix_path_captured_latent_code {path_testing_latent_code}')
    os.system(f'torchrun --nproc_per_node=1 --master_port=25678  test.py --data_dir {path_prefix}/dataset/m--20180227--0000--6795937--GHS --krt_dir {path_prefix}/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test {path_prefix}/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path "/home/jianming/work/Privatar_prj/training_results/horizontal_partition_{BDCT_thres}/best_model.pth" --frequency_threshold {BDCT_thres} --arch base --average_texture_path {path_prefix}{average_texture_path} --prefix_path_captured_latent_code {path_testing_latent_code}')


