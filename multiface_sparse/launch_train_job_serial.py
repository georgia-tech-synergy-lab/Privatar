import os

# Please select one path out of the following three paths based on the server
path_prefix_god2 = "/home/jianming/work/multiface/"
path_prefix_synergy3 = "/home/jianming/work/Privatar_prj/training_results/"
path_prefix_uwing2 = "/workspace/uwing2/multiface/"

# Please select one path out of the following three paths based on the server
#######
path_prefix = path_prefix_god2
#######

sparsity_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# On syenrgy3 machine --- the following codes should be executed
# scl enable devtoolset-11 bash
#  --master_port=25678
for sparsity in sparsity_list:
    if not os.path.exists(f"/home/jianming/work/Privatar_prj/training_results/sparse_0_{str(sparsity).split('.')[-1]}"):
        os.makedirs(f"/home/jianming/work/Privatar_prj/training_results/sparse_0_{str(sparsity).split('.')[-1]}")
    print(f'torchrun --master_port=25678 --nproc_per_node=1 train.py --data_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS --krt_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path /home/jianming/work/Privatar_prj/training_results/sparse_0_{str(sparsity).split(".")[-1]} --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt "/home/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth" --arch base --unified_pruning_ratio {sparsity} --project_name chnl_sparsity --author_name jimmytong')
    os.system(f'torchrun --master_port=25678 --nproc_per_node=1 train.py --data_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS --krt_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path /home/jianming/work/Privatar_prj/training_results/sparse_0_{str(sparsity).split(".")[-1]} --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt "/home/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth" --arch base --unified_pruning_ratio {sparsity} --project_name chnl_sparsity --author_name jimmytong')
