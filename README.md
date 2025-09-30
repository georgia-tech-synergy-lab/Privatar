# Privatar: Enabling Privacy-Preserving Real-Time Multi-Users VR Through Secure Outsourcing
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)


# What's Privatar?
Privatar is the first that leverages both local and untrusted cloud to concurrently achieve privacy-preserving multi-user avatar reconstruction. The entire post-split flow is illustrated as the figure below.

<img src="figure/setup.png" width="1400">



# Structure of the Repo: 
- multiface_partition: Adopt BDCT filter with block=8, which gives 64 different frequency blocks, this does not have having input upsampling and remove three convolution layers.
- multiface_partition_bdct4x4: Decompose unwrapped texture into multiple frequency components -- Adopt BDCT filter with block=4, which gives 16 different frequency blocks, this remove 2 convolution layers, no outsourcing.
- multiface_partition_bdct4x4_hp: Decompose unwrapped texture into multiple frequency components AND horizontally partition all components into local and cloud. Adopt BDCT filter with block=4, which gives 16 different frequency blocks, this remove 2 convolution layers. The number of offloaded frequency components is controlled via ```num_freq_comp_offloaded```.
- multiface_sparse: add sparsity to only decoder of the original VAE model.
- multiface_quantization: change the bitprecision of data into 8-/16-/32-bit integer for the decoder only.
- multiface_direct_split: directly split the model architecture into private and public branches.


# Installation
We list two flows to install, run and test Privatar because different environments have different package in support. E.g. RTX 3090 supports ```RasterizeGLContext``` but GH200 only supports ```RasterizeCudaContext```. To reduce the effort for setting up configurations for different platforms, we directly prepare two sets of scripts, separately. 

## Step 1: Setup GPU Docker
<!-- -  For typical desktop-class GPU such as RTX 3090, we recommend using the conda virtual environment.
```bash
conda create -n <your_favoriate_name> python=3.7
conda activate <your_favoriate_name>
``` 

- For cloud-class GPU such as GH200 
We recommand using NVIDIA built-in docker.-->
We recommand using NVIDIA built-in docker.

Command 1: Download the docker.
If you have Docker 19.03 or later, a typical command to launch the container is:
```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:xx.xx-py3
```
Note: our experiments base on nvcr.io/nvidia/pytorch:24.01-py3

Command 2: Download this repo to <path>
```bash
git clone https://github.com/georgia-tech-synergy-lab/Privatar.git
```

Command 3: Launch the docker which links <path> to `/work` in docker.
```bash
docker run --gpus all -v <path>:/work -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --memory 51200m  --rm <docker_name>
```
where <docker_name> refers to the name of your downloaded docker "nvcr.io/nvidia/pytorch:xx.xx-py3", and <path> refers to the path to Privatar.

**CRITICAL! All following scripts assuming the `/work` is the path to this repo in docker**

## Step 2: Install Required Dependency
Within in the Docker, install required dependency.
- Install OS-level dependencies
```bash
$ apt-get install mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev
$ apt-get install mesa-utils
$ glxinfo | grep -i opengl
```

- Install python packages
```bash
pip3 install torch
pip3 install Pillow ninja imageio imageio_ffmpeg six tensorboard opencv-python==4.8.0.74 wandb torchjpeg
pip3 install -U opencv-python
```
Note: 
1. we use "wandb" to track the training, testing progress and record the final results.
By default, wandb is turned off, u could change `wandb_enable` from each training/testing script to enable wandb.
2. If the GPU is 5090, `pip install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130` should be used instead of `pip3 install torch`.


- Install nvdiffrast package
```bash
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
python3 setup.py install
```

Note: if u are using RTX5090, please run the following patch after installing nvdiffrast to enable the feature.

```bash
source /work/experiment_scripts/nvdiffrast_patch.sh
```

<!-- - Download and install torchjpeg (if )
```bash
git clone https://github.com/Queuecumber/torchjpeg
cd torchjpeg
pip3 setup.py install
```
We don't recommend installing torchjpeg using pip3 install because it might break other dependencies. -->

## Step 3: Download Datasets
```bash
cd /work
mkdir dataset
python3 ./multiface/download_dataset.py --dest "/work/dataset" --download_config "./mini_download_config.json"
```
If u follow above instructions, then /work should be `/work`.

## Step 4: Download pretrained model
The pretrained weights for different users in the provided datasets are collected at this [facial_pretrained_datasets](https://github.com/facebookresearch/multiface/blob/main/documentation/INSTALLATION.md). We use 6795937 base model as the evaluation target. Other structure would work as well. The link for the pretrained model weights of 6795937 is [6795937_base](https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/PretrainedModel/6795937--GHS-base_nosl/best_model.pth)

```bash
cd /work
mkdir pretrain_model
cd pretain_model
wget https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/PretrainedModel/6795937--GHS-base_nosl/best_model.pth -O 6795937_best_model.pth
```

# Ready to run?
## Step 1: Training
- Original: multiface baseline using the pretrained model weights
```bash
cd /work
git clone https://github.com/facebookresearch/multiface.git
cd multiface
python3 launch_train_job_serial.py
```

- Design Choice 1: directly split mesh and unwrapped texture into two separate path (offload entire unwrapped texture)
```bash
cd /work/multiface_direct_split
python3 launch_train_job_serial.py
```

- Design Choice 2: quantize model to be low precision
```bash
cd /work/multiface_quantization
python3 launch_train_job_serial.py
```

- Design Choice 3: prune channels from the decoder to reduce the local computation
```bash
cd /work/multiface_sparse
python3 launch_train_job_serial.py
```

- Design Choice 4: decompose "unwrapped texture" into 16 frequency components, but keep all of them run local.
This requires training to be completed.
```bash
cd /work/multiface_frequency_decompose
python3 launch_train_job_serial.py
```

- Design Choice 5: decompose "unwrapped texture" into 16 frequency components, configurable components outsourcing. The number of frequency components are controlled by ```num_freq_comp_offloaded```.
This requires training to be completed.
```bash
cd /work/multiface_partition_frequency_decompose
python3 launch_train_job_serial.py
```

Note: all training results locate at the ```/work/training_results``` folder.


## Step 2: Testing
Note that these test directly iterate the whole test dataset in random orders.
- Original: multiface baseline using the pretrained model weights
```bash
cd /work/multiface
python3 launch_test_job_serial.py
```

- Design Choice 1: directly split mesh and unwrapped texture into two separate path (offload entire unwrapped texture)
```bash
cd /work/multiface_direct_split
python3 launch_test_job_serial.py
```

- Design Choice 2: quantize model to be low precision
```bash
cd /work/multiface_quantization
python3 launch_test_job_serial.py
```

- Design Choice 3: prune channels from the decoder to reduce the local computation
```bash
cd /work/multiface_sparse
python3 launch_test_job_serial.py
```

- Design Choice 4: decompose "unwrapped texture" into 16 frequency components, but keep all of them run local.
This requires training to be completed.
```bash
cd /work/multiface_frequency_decompose
python3 launch_test_job_serial.py
```

- Design Choice 5: decompose "unwrapped texture" into 16 frequency components, configurable components outsourcing. The number of frequency components are controlled by ```num_freq_comp_offloaded```.
This requires training to be completed.
```bash
cd /work/multiface_partition_frequency_decompose
python3 launch_test_job_serial.py
```

Note: all testing results locate at the ```/work/testing_results``` folder.

## Step 3: Testing (For designated expressions)
we also provide following scripts in case you only wanna see how the model performs for a specific set of images. Specifically, put the original path of the original images in the dataset into the file `/work/experiment_scripts/render_scripts/test_image_path`. And then run following commands.

The input images could be generated by run following script.
```bash
python3 /work/experiment_scripts/render_scripts/render_test_expression.py
```
The ground truth of input images would be written to `/work/render_results/ground_truth_input_testing_data`.

- Original: multiface baseline using the pretrained model weights
```bash
cd /work/multiface
python3 launch_test_selected_expressions.py
```

- Design Choice 1: directly split mesh and unwrapped texture into two separate path (offload entire unwrapped texture)
```bash
cd /work/multiface_direct_split
python3 launch_test_selected_expressions.py
```

- Design Choice 2: quantize model to be low precision
```bash
cd /work/multiface_quantization
python3 launch_test_selected_expressions.py
```

- Design Choice 3: prune channels from the decoder to reduce the local computation
```bash
cd /work/multiface_sparse
python3 launch_test_selected_expressions.py
```

- Design Choice 4: decompose "unwrapped texture" into 16 frequency components, but keep all of them run local.
This requires training to be completed.
```bash
cd /work/multiface_frequency_decompose
python3 launch_test_selected_expressions.py
```

- Design Choice 5: decompose "unwrapped texture" into 16 frequency components, configurable components outsourcing. The number of frequency components are controlled by ```num_freq_comp_offloaded```.
This requires training to be completed.
```bash
cd /work/multiface_partition_frequency_decompose
python3 launch_test_selected_expressions.py
```

Note that, we use `/work/dataset/m--20180227--0000--6795937--GHS/images/E009_Smile_Mouth_Open/400023/002799.png` (representing high dynamic expression) and `/work/dataset/m--20180227--0000--6795937--GHS/images/E001_Neutral_Eyes_Open/400009/000102.png` (representing low dynamic expression) to obtain the figure 1 in paper.

## Step 4: Latency Profiling
we use `torch.jit.trace` to optimize the kernel offloaded to GPU and pick which ever one between traced version and untraced version to give better performance.

- Original: multiface baseline using the pretrained model weights
```bash
cd /work/multiface
python3 latency_profiling_script.py
```

- Design Choice 2: quantize model to be low precision
```bash
cd /work/multiface_quantization
python3 latency_profiling_script.py
```
Note: change `bitwidth=<val>` in `model = decoder_linear_quantization(model, bitwidth=8, datatype=torch.int8)` to change its precision.

- Design Choice 3: prune channels from the decoder to reduce the local computation
```bash
cd /work/multiface_sparse
python3 latency_profiling_script.py
```

- Design Choice 4: decompose "unwrapped texture" into 16 frequency components, but keep all of them run local.
This requires training to be completed.
```bash
cd /work/multiface_frequency_decompose
python3 latency_profiling_script.py
```

- Design Choice 5: decompose "unwrapped texture" into 16 frequency components, configurable components outsourcing. The number of frequency components are controlled by ```num_freq_comp_offloaded```.
This requires training to be completed.
```bash
cd /work/multiface_partition_frequency_decompose
python3 latency_profiling_script_local_path.py
python3 latency_profiling_script_offload_path.py
```
`latency_profiling_script_local_path.py` measures latency needed for local path, which runs locally on VR headset.
`latency_profiling_script_offload_path.py` measures latency needed for offloaded path, which runs on untrusted devices (a PC with GPU here).

To further understand the total amount of computation, we also offer a script to compute the Flops needed for both paths under different configurations.

```bash
python3 latency_flops_calculation.py
```

Note: `multiface_direct_split` is a design choice for understanding the information, hence we don't give latency profiling script for it.

## Step 5: Noise Calculation
Before calculating noise, two following steps need to be completed.
1. **Complete Training**: Train the model to generate the final weights required for testing.

2. **Perform Testing**: Use the trained model weights to evaluate. 
During testing, the latent codes are stored in the following directory:
   - `./testing_results/<project_name>/latent_code/`
     - `z_<id>.pth`: Latent code for the local path.
     - `z_offload_<id>.pth`: Latent code for the offloaded path (runs on the cloud). 
Note: the `save_latent_code` command in the script `launch_test_job_serial.py` controls 
whether to explicitlys store latent code to the external drive.

### Calculate Differential Privacy based Noise
Differential Privacy (DP) noise calculation follows [paper](https://arxiv.org/pdf/1702.07476). 
DP requires the prior knowledge of the L2 norm of all offloaded latent codes.
Detailed procedures are written as the comments in the code.

```bash
cd /work/experiment_scripts/dp_analysis
python3 dp_noise_generation_for_multiface.py
```
This `dp_noise_generation_for_multiface.py` calculates the amount of DP-based noises needed to protect information when using **completed offload**, i.e. the entire decoder of the original multiface is offloaded.


```bash
cd /work/experiment_scripts/dp_analysis
python3 dp_noise_generation_for_partition_multiface.py
```
The `dp_noise_generation_for_partition_multiface.py` script conducts two partitioned design choice.
1. `multiface_direct_split`, which directly offloads the entire unwrapped texture. This is the default choice as it's baseline (completed offloaded + DP noise).
2. `multiface_partition_frequency_decompose`, which offloads selected high frequency components of unwrapped texture.

We pre-pick posterious successful rate to be [0.98, 0.827, 0.4, 0.09, 0.035], means there potentially exists an attack who could mount attack with possibility of [98%, 82.7%, 40%, 9%, 3.5%]. This boils down to mutual information list as [4, 3, 1, 0.1, 0.01].

After this script, the generated noise will show up in the path 
`/work/experiment_scripts/dp_analysis/generated_dp_noise/dp_noise_partition_<num_offloaded_freq_components>_<mutual_information_bound>.npy`

### Calculate PAC noise
Noise calculation following PAC privacy requires the prior knowledge of the L2 norm of **the covariance** of all offloaded latent codes.
So that PAC privacy could leverage the dimensional differences to generate non-uniform noise for minimizing the overall noise intensity.

```bash
python3 /work/experiment_scripts/pac_analysis/pac_noise_generation_for_partition_multiface.py
```
After this script, the generated noise will show up in the path 
`/work/experiment_scripts/pac_analysis/noise_covariance/pac_noise_partition_<num_offloaded_freq_components>_<mutual_information_bound>.npy`

For generating PAC noise, we directly use the same mutual information as DP which gives the same provable privacy guarantee as DP noise generation.

## Step 6: Noisy Inference
After noise calculation, now we could start testing the actual accuracy and latency 
of noisy inference! Specifically, in horizontally partitioned avatar reconstruction 
flow, generated noise is only injected to the offloaded latent codes.

Noisy inference
```bash
cd /work/multiface_partition_frequency_decompose
python3 launch_noisy_test_job_serial.py
```
In this script, the noise generated from Step 5 will be fed in as `gaussian_noise_covariance_path` to the model.
This generates detailed accuracy of noisy horizontal partitioned avatar reconstruction 
under various partition configurations.


Beyond the proposed horizontal frequency based partitioned design choice, we also offer 
noisy inference code for directly split design case.


## Step 7: Empirical Attack
The attacker guesses the expression with the minimal difference through comparing the estimated frequency component to the reference expression, as shown by the Fig. 14 of the paper. Therefore, we need to first define the frequency components of reference expressions.

We provide three different ways of generating reference components, under all of which the attacker shows similar empirical successful rate.

- Method 1: ```accumulate_channel``` mode, where only high frequency components are used as reference. 
- Method 2: ```attack_from_high_frequency_channel``` mode, where only high frequency components are used as reference. 
- Method 3: when both modes are set as False, all 16 frequency components are sorted based on the amibuigity, and the frequency components coming with similar amount of ambuiguity will be merged as one ```if model.normalize_list[freq_pair[0]] + model.normalize_list[freq_pair[1]] < 2:```, where 2 is a random choice that could be changed into different values for obtaining different merging strategies.

Specify the configuration of ```accumulate_channel``` and ```attack_from_high_frequency_channel```, then run the following script to launch attack.

```bash
python3 launch_empirical_attack.py
```

After reference frequency components are set, we run an empirical identification attack against a pretrained DeepAppearanceVAE_Partition model by matching predicted high-frequency texture components to precomputed components from each expression, and reports the identification accuracy.

The final accuracy will be printed out as ```attack_accuracy_mean <final_PSR>```

Note that: ```using_pac_noise``` is the command to control using PAC privacy based noise or Differential Privacy based noise.

## Step 8: Mount NN based Attack

We also train a three-layer fully connected network which estimates the expression from the offloaded noisy latent code. Run following script to start the training.

The NN attacker randomly takes one sample from each expression, as detailed in `/work/experiment_scripts/empirical_attack/selected_expression_frame_list.txt`, which becomes its training dataset.

```bash
cd /work/multiface_partition_frequency_decompose
python3 launch_train_nn_attacker.py
```

After training, launch the attack via
```bash
cd /work/multiface_partition_frequency_decompose
python3 launch_test_nn_attacker.py
```

# Additional Scripts
We also offer few scripts for researchers, who are interested in exploring the model more deeply, to see different statistics and configurations.

## Script 1: BDCT reconstruction script
```/work/experiment_scripts/bdct_reconstruction/bdct_4x4_reconstruction_dataloader.ipynb``` includes the script to decomposes given unwrapped texture into different frequency components, and list few different configurations of decomposition.

For each configuration, all frequency components will be written to the folder ```/work/experiment_scripts/bdct_reconstruction``` for visualization.

## Script 2: Analyze the covariance of each frequency component for different datasets
To help understand the covariance of different frequency components under differnet datasets, we provide the script to perform running profiling of all decomposed frequency components under designated datasets. 

To run it,
```bash
cd /work/multiface_frequency_decompose
python3 launch_l2norm_freq_cov_analysis.py
```

It will shows results like following
```bash
trace of covariance = 11308.309006199575 for freq component = 0
trace of covariance = 199.41605765586263 for freq component = 1
trace of covariance = 77.53071010274161 for freq component = 2
trace of covariance = 41.89610293635083 for freq component = 3
trace of covariance = 152.33769320529836 for freq component = 4
trace of covariance = 33.72750777014488 for freq component = 5
trace of covariance = 26.97710811919577 for freq component = 6
trace of covariance = 19.355714181792997 for freq component = 7
trace of covariance = 38.930325425557726 for freq component = 8
trace of covariance = 25.547661178709628 for freq component = 9
trace of covariance = 23.260128349715654 for freq component = 10
trace of covariance = 18.580644217635967 for freq component = 11
trace of covariance = 23.134909910006407 for freq component = 12
trace of covariance = 16.6648382626215 for freq component = 13
trace of covariance = 16.267422970259233 for freq component = 14
trace of covariance = 12.812623106426202 for freq component = 15
```

Note, `custom_scripts` contains scripts for development, maintainence and verification purpose.


## Script 3: Render a specific expression or a selected sets of expression using well-trained model.
To help creating visualized expression rendered from a given model configuration, we also offer script under each setup to render visual avatar prediction for specified input images.

Specifically, the set of images the model would render is detailed ```/work/experiment_scripts/render_scripts/test_image_path```.

To launch the image rendering,
```bash
cd /work/<path_to_configuration>/
python3 launch_test_all_expressions_RTX3090.py
```
where ```<path_to_configuration>``` could be multiface, multiface_frequency_decompose, multiface_quantization, multiface_direct_split, multiface_partition_frequency_decompose and multiface_sparse.

It will shows results into the folder ```/work/render_results/<configuration_name>```


Have Fun! Enjoy! :D
