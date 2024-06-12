# Training script
torchrun --nproc_per_node=1 train.py --data_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS --krt_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path "/home/jianming/work/Privatar_prj/training_results" --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt "/home/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth" --arch base


# Testing script #test samples 44536
torchrun --nproc_per_node=1 test.py --data_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS --krt_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path "/home/jianming/work/Privatar_prj/testing_results" --test_segment "./test_segment.json" --lambda_screen 1 --model_path "/home/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth" --arch base


# Visualization script #test samples 44536 -- could use for testing
torchrun --nproc_per_node=1 visualize.py --data_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS --krt_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path "/home/jianming/work/Privatar_prj/visual_results" --test_segment_config "./test_segment.json" --lambda_screen 1 --model_path "/home/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth" --arch base --camera_config /home/jianming/work/Privatar_prj/multiface/camera_configs/camera-split-config_6795937.json --camera_setting "full"


# Testing script -- partitioned model #test samples 44536
torchrun --nproc_per_node=1 --master_port=25678  test.py --data_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS --krt_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path "/home/jianming/work/Privatar_prj/testing_results/horizontal_partition_4.0" --test_segment "./test_segment.json" --lambda_screen 1 --model_path "/home/jianming/work/Privatar_prj/training_results/horizontal_partition_0.4/best_model.pth" --frequency_threshold 0.4 --arch base


# Testing script -- partitioned model #test samples 44536 -- uwing2
torchrun --nproc_per_node=1 --master_port=25680  test_uwing2.py --data_dir /workspace/uwing2/multiface/dataset/m--20180227--0000--6795937--GHS --krt_dir /workspace/uwing2/multiface/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test /workspace/uwing2/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path /workspace/uwing2/Privatar/partition_results/horizontal_partition_0.4 --test_segment "./test_segment.json" --lambda_screen 1 --model_path "/workspace/uwing2/Privatar/partition_results/horizontal_partition_0.4/best_model.pth" --average_texture_path "/workspace/uwing2/multiface/dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average/000102.png" --frequency_threshold 0.4 --arch base

