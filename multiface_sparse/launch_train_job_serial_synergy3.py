import os

# Please select one path out of the following three paths based on the server
path_prefix_god2 = "/home/jianming/work/multiface/"
path_prefix_synergy3 = "/usr/scratch/jianming/Privatar/training_results/"
path_prefix_uwing2 = "/workspace/uwing2/multiface/"

# Please select one path out of the following three paths based on the server
#######
path_prefix = path_prefix_synergy3
#######

sparsity_list = [0.6]

# On syenrgy3 machine --- the following codes should be executed
# scl enable devtoolset-11 bash
#  --master_port=25678
device_id = 1
port_id = 25680
for sparsity in sparsity_list:
    if not os.path.exists(f"/usr/scratch/jianming/Privatar/training_results/sparse_0_{str(sparsity).split('.')[-1]}"):
        os.makedirs(f"/usr/scratch/jianming/Privatar/training_results/sparse_0_{str(sparsity).split('.')[-1]}")
    print(f'CUDA_VISIBLE_DEVICES={device_id} torchrun --master_port={port_id} --nproc_per_node=1 train.py --data_dir /usr/scratch/jianming/multiface/dataset/m--20180227--0000--6795937--GHS --krt_dir /usr/scratch/jianming/multiface/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train /usr/scratch/jianming/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test /usr/scratch/jianming/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path /usr/scratch/jianming/Privatar/training_results/sparse_0_{str(sparsity).split(".")[-1]} --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt "/usr/scratch/jianming/multiface/pretrained_model/6795937_best_base_model.pth" --arch base --unified_pruning_ratio {sparsity} --project_name chnl_sparsity --author_name jimmytong')
    os.system(f'CUDA_VISIBLE_DEVICES={device_id} torchrun --master_port={port_id} --nproc_per_node=1 train.py --data_dir /usr/scratch/jianming/multiface/dataset/m--20180227--0000--6795937--GHS --krt_dir /usr/scratch/jianming/multiface/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train /usr/scratch/jianming/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test /usr/scratch/jianming/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path /usr/scratch/jianming/Privatar/training_results/sparse_0_{str(sparsity).split(".")[-1]} --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt "/usr/scratch/jianming/multiface/pretrained_model/6795937_best_base_model.pth" --arch base --unified_pruning_ratio {sparsity} --project_name chnl_sparsity --author_name jimmytong')
