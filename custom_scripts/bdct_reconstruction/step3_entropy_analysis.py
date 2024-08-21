import torch

run_mode = "all" # "all"

# wanna to decide the arrange of channel by the information.
"""
    From a subset of the model
"""
if run_mode == "subset":
    l2_norm_drop_freq_difference_array = []
    for freq_id in range(15):
        l2_norm_drop_freq_difference_array.append(torch.norm(torch.load(f"/home/jianming/work/Privatar_prj/custom_scripts/bdct_reconstruction/l2_norm_drop_freq_difference_array_freq_comp_{freq_id}.pth")))

    print(l2_norm_drop_freq_difference_array)

# [tensor(9538193.), tensor(2152726.5000), tensor(1344386.3750), tensor(1019910.9375), tensor(1808491.5000), tensor(1293464.5000), tensor(1079083.6250), tensor(926193.5625), tensor(1157143.7500), tensor(1034297.8125), tensor(914815.7500), tensor(813751.1875), tensor(930733.1250), tensor(869063.4375), tensor(809124.5000)]

"""
    From the entire model
"""
if run_mode == "all":
    all_expression_l2_norm_drop_freq_difference_array = []
    for freq_id in range(15):
        all_expression_l2_norm_drop_freq_difference_array.append(torch.norm(torch.load(f"/home/jianming/work/Privatar_prj/custom_scripts/bdct_reconstruction/all_expression_l2_norm_drop_freq_difference_array_freq_comp_{freq_id}.pth")))

    print(all_expression_l2_norm_drop_freq_difference_array)
