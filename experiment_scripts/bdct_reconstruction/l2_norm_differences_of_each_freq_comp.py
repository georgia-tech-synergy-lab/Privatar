from library import *
from tqdm import tqdm
import sys

l2_norm_list = []

total_freq_component = 16
freq_comp = [[] for _ in range(total_freq_component)]
for i, data in tqdm(enumerate(test_loader)):
    texture_in = data["avg_tex"].to("cpu")
    bs, ch, h, w = texture_in.shape
    block_imgs = dct_transform(texture_in, bs, ch, h, w)
    for frequency_id in range(total_freq_component):
      freq_comp[frequency_id].append(block_imgs[0, frequency_id, :, :, :].to("cuda:0"))

for frequency_id in range(total_freq_component):
    overall_components = np.zeros((len(freq_comp[frequency_id]), len(freq_comp[frequency_id])))

    l2_norm_drop_freq_difference_array = torch.zeros(len(freq_comp[frequency_id]), len(freq_comp[frequency_id]))
    for i, freq_data_pair1 in tqdm(enumerate(freq_comp[frequency_id])):
        for j, freq_data_pair2 in enumerate(freq_comp[frequency_id]):
            l2_norm_drop_freq_difference_array[i,j] = torch.norm(torch.subtract(freq_data_pair1, freq_data_pair2))

    torch.save(l2_norm_drop_freq_difference_array, f"all_expression_l2_norm_drop_freq_difference_array_freq_comp_{frequency_id}.pth")
    l2_norm_list.append(torch.norm(l2_norm_drop_freq_difference_array))

print(f"l2 norm list of all frequency components = {l2_norm_list}")