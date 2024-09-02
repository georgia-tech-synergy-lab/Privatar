from library import *
from tqdm import tqdm
import sys

if __name__ == "__main__":
    texture_list = []
    for i, data in tqdm(enumerate(test_loader)):
        texture_list.append(data["avg_tex"].to("cpu"))

    l2_norm_drop_freq_difference_array = torch.zeros(len(texture_list), len(texture_list))
    for i, freq_data_pair1 in tqdm(enumerate(texture_list)):
        for j, freq_data_pair2 in enumerate(texture_list):
            l2_norm_drop_freq_difference_array[i,j] = torch.norm(torch.subtract(freq_data_pair1, freq_data_pair2))
    
    torch.save(l2_norm_drop_freq_difference_array, f"all_expression_l2_norm_drop_freq_difference_array_overall.pth")
