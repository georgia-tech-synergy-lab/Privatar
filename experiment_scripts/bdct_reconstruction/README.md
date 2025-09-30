# Overview
This repo is designed in helping 

## Input Frequency Filter Exploration Overview
Privatar breaks input unwrapped texture into many frequency components. This folder implements the script to break both (1) raw unwrapped texture [notebook1] and (2) the difference between unwrapped texture and texture average (`/work/dataset/m--20180227--0000--6795937--GHS/unwrapped_uv_1024/E001_Neutral_Eyes_Open/average`) into 16 frequency components and then reconstruct these 16 frequency components back to the original input image [notebook2].

Further, this folder also implements following logics to drop frequency components and then see how it affects the difference of reconstructed image and original image.
1. drop high frequency components by masking them to all zeros, e.g. forcing higher 2 frequency components to zeros
2. drop low frequency components by masking them to all zeros, e.g. forcing lower 2 frequency components to zeros

- [notebook1] `/work/experiment_scripts/bdct_reconstruction/bdct_4x4_reconstruction_dataloader.ipynb`
- [notebook2] `/work/experiment_scripts/bdct_reconstruction/bdct_4x4_reconstruction_raw_img.ipynb`

Both notebooks prove following conclusions
1. zeroing out higher frequency is better than using duplicating lower frequency to higher frequency components.
2. The same conclusion apply to both raw image and difference between raw image (input unwrapped texture) and average texture.

## L2 norm of each Frequency Component
In order to quantatively measure the information contained in each frequency components, we offer a script `l2_norm_differences_of_each_freq_comp.py` to measure the L2 norm of differences for each individual frequency component.

Further, image blurring is another approach which is often used in spliting input image into multiple components, as specified in the `blur_filter/l2_norm_blur_filter.py`

