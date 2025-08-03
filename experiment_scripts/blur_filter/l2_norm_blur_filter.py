from library import *
from tqdm import tqdm
import sys


def apply_filters(pytorch_tensor):
    # Convert pytorch_tensor into format understanable by openCV
    img = pytorch_tensor.permute(1, 2, 0).cpu().numpy()

    # Check if the image is loaded properly
    if img is None:
        raise ValueError("The image could not be loaded. Please check the file path.")

    # Apply Box Filter (Blurring)
    box_blur = cv2.blur(img, (30, 30))

    # convert results back to torch tensor.
    result_tensor = torch.from_numpy(box_blur).permute(2, 0, 1)  # Convert back to C x H x W format
    return result_tensor
    


if __name__ == "__main__":
    blurred_comp = []
    # obtain the blur data library
    for i, data in tqdm(enumerate(test_loader)):
        texture_in = data["avg_tex"].to("cpu")
        bs, ch, h, w = texture_in.shape
        box_blur = apply_filters(texture_in[0,:,:,:])
        blurred_comp.append(box_blur)

    l2_norm_drop_freq_difference_array = torch.zeros(len(blurred_comp), len(blurred_comp))
    for i, freq_data_pair1 in tqdm(enumerate(blurred_comp)):
      for j, freq_data_pair2 in enumerate(blurred_comp):
          l2_norm_drop_freq_difference_array[i,j] = torch.norm(torch.subtract(freq_data_pair1, freq_data_pair2))
  
    torch.save(l2_norm_drop_freq_difference_array, f"blur_norm.pth")
