from library import *
from tqdm import tqdm
import sys

frequency_id = int(sys.argv[1])

if __name__ == "__main__":
    freq_comp = []
    for i, data in tqdm(enumerate(test_loader)):
        texture_in = data["avg_tex"].to("cpu")
        bs, ch, h, w = texture_in.shape
        block_imgs = dct_transform(texture_in, bs, ch, h, w)
        freq_comp.append(block_imgs[frequency_id, :, :, :, :].to("cuda:0"))
    freq_comp[0].flatten().shape[0]
    
    overall_components = np.zero(len(freq_comp), len())

    l2_norm_drop_freq_difference_array = torch.zeros(len(freq_comp), len(freq_comp))
    for i, freq_data_pair1 in tqdm(enumerate(freq_comp)):
        for j, freq_data_pair2 in enumerate(freq_comp):
            l2_norm_drop_freq_difference_array[i,j] = torch.norm(torch.subtract(freq_data_pair1, freq_data_pair2))
    
    torch.save(l2_norm_drop_freq_difference_array, f"all_expression_l2_norm_drop_freq_difference_array_freq_comp_{frequency_id}.pth")
