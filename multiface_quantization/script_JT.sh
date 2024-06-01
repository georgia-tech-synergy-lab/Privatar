# Training script
torchrun --nproc_per_node=1 train.py --data_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS --krt_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_train /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt --framelist_test /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path "/home/jianming/work/Privatar_prj/training_results" --test_segment "./test_segment.json" --lambda_screen 1 --model_ckpt "/home/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth" --arch base


# Testing script #test samples 44536
torchrun --nproc_per_node=1 test.py --data_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS --krt_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path "/home/jianming/work/Privatar_prj/testing_results" --test_segment "./test_segment.json" --lambda_screen 1 --model_path "/home/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth" --arch base


# Visualization script #test samples 44536 -- could use for testing
torchrun --nproc_per_node=1 visualize.py --data_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS --krt_dir /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/KRT --framelist_test /home/jianming/work/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt  --result_path "/home/jianming/work/Privatar_prj/visual_results" --test_segment_config "./test_segment.json" --lambda_screen 1 --model_path "/home/jianming/work/multiface/pretrained_model/6795937_best_base_model.pth" --arch base --camera_config /home/jianming/work/Privatar_prj/multiface/camera_configs/camera-split-config_6795937.json --camera_setting "full"
