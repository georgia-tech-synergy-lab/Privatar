# Training script
torchrun --nproc_per_node=1 train.py --data_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS --krt_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path "/home/jianming/work/Privatar_prj/training_results/test" --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt "/home/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth" --arch base


# Testing script #test samples 44536
torchrun --nproc_per_node=1 test.py --data_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS --krt_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path "/home/jianming/work/Privatar_prj/testing_results" --test_segment "./test_segment.json" --lambda_screen 1 --model_path "/home/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth" --arch base


# Visualization script #test samples 44536 -- could use for testing
torchrun --nproc_per_node=1 visualize.py --data_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS --krt_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path "/home/jianming/work/Privatar_prj/visual_results" --test_segment_config "./test_segment.json" --lambda_screen 1 --model_path "/home/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth" --arch base --camera_config /home/jianming/work/Privatar_prj/multiface/camera_configs/camera-split-config_6795937.json --camera_setting "full"


# Training script -- sparsity
torchrun --nproc_per_node=1 --master_port=25678 train.py --data_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS --krt_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path "/home/jianming/work/Privatar_prj/training_results/sparse_0_2" --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt "/home/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth" --arch base --unified_pruning_ratio 0.2 --project_name chnl_sparsity --author_name jimmytong


# Visualization script #test samples 44536 -- could use for testing -- sparsity
torchrun --master_port=25678 --nproc_per_node=1 visualize.py --data_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS --krt_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path "/home/jianming/work/Privatar_prj/visual_results/sparse_0_1" --test_segment_config "./test_segment.json" --lambda_screen 1 --model_path "/home/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth" --pruning_model_path "/home/jianming/work/Privatar_prj/training_results/sparse_0_1/training_results_0_1/best_model.pth" --arch base --camera_config /home/jianming/work/Privatar_prj/multiface/camera_configs/camera-split-config_6795937.json --camera_setting "full" 


# Training script -- applying quantization
torchrun --master_port=25678 --nproc_per_node=1 train.py --data_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS --krt_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path "/home/jianming/work/Privatar_prj/training_results/quantization" --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt "/home/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth" --arch base

# Training script -- horizontal partitioning
torchrun --nproc_per_node=1 --master_port=25678 train.py --data_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS --krt_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path "/home/jianming/work/Privatar_prj/training_results/horizontal_partition_0_4" --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt "/home/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth" --arch base --frequency_threshold 0.4  --project_name chnl_sparsity --author_name jimmytong

