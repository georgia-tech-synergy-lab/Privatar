import os

# Please select one path out of the following three paths based on the server
data_dir = "/work/dataset"

for num_freq_comp_offloaded in [2, 4, 6, 8, 10, 12, 14]:
  best_model_path = f"/work/training_results/partition_{num_freq_comp_offloaded}/best_model.pth"
  framelist_test = "/work/experiment_scripts/empirical_attack/selected_expression_frame_list.txt"
  camera_configs_path = "/work/experiment_scripts/empirical_attack/attack-camera-split-config_6795937.json"
  epoch = 10
  input_feature = 256
  hidden_feature = 128
  output_feature = 65
  lr = 1e-4
  momentum  = 0
  nn_attacker_model_path = f"/work/training_results/nn_attacker_{num_freq_comp_offloaded}/best_attacker_model.pth"

  using_pac_noise = True

  # training nn attacker
  result_path_prefix = "/work/training_results/"
  project_name = f"nn_attacker_{num_freq_comp_offloaded}"
  result_path = f"{result_path_prefix}{project_name}"

  print(f'python train_nn_attacker.py --data_dir {data_dir}/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/m--20180227--0000--6795937--GHS/KRT --framelist_test {framelist_test}  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong --camera_configs_path {camera_configs_path} --epoch {epoch} --input_feature {input_feature} --hidden_feature {hidden_feature} --output_feature {output_feature}  --lr {lr} --momentum {momentum}')
  os.system(f'python train_nn_attacker.py --data_dir {data_dir}/m--20180227--0000--6795937--GHS --krt_dir {data_dir}/m--20180227--0000--6795937--GHS/KRT --framelist_test {framelist_test}  --result_path {result_path} --test_segment "./test_segment.json" --lambda_screen 1 --model_path {best_model_path}  --arch base  --project_name {project_name} --author_name jimmytong  --num_freq_comp_offloaded {num_freq_comp_offloaded} --camera_configs_path {camera_configs_path} --epoch {epoch} --input_feature {input_feature} --hidden_feature {hidden_feature} --output_feature {output_feature}  --lr {lr} --momentum {momentum}')
