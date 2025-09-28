# Privatar: Enabling Privacy-Preserving Real-Time Multi-Users VR Through Secure Outsourcing
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)


# What's Privatar?
Privatar is the first that leverages both local and untrusted cloud to concurrently achieve privacy-preserving multi-user avatar reconstruction. The entire post-split flow is illustrated as the figure below.

<img src="figure/setup.png" width="1400">



# Structure of the Repo: 
- multiface_partition: Adopt BDCT filter with block=8, which gives 64 different frequency blocks, this does not have having input upsampling and remove three convolution layers.
- multiface_partition_bdct4x4: Decompose unwrapped texture into multiple frequency components -- Adopt BDCT filter with block=4, which gives 16 different frequency blocks, this remove 2 convolution layers, no outsourcing.
- multiface_partition_bdct4x4_hp: Decompose unwrapped texture into multiple frequency components AND horizontally partition all components into local and cloud. Adopt BDCT filter with block=4, which gives 16 different frequency blocks, this remove 2 convolution layers. The number of outsourced frequency components is controlled via ```num_freq_comp_outsourced```.
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

## Step 2: Install Required Dependency
Within in the Docker, install required dependency.
- Install OS-level dependencies
```bash
$ apt-get install mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev
$ apt-get install mesa-utils
$ glxinfo | grep -i opengl
```

- Install python packages
```
pip3 install torch
pip3 install Pillow ninja imageio imageio_ffmpeg six tensorboard opencv-python==4.8.0.74 wandb torchjpeg
```
Note: we use "wandb" to track the training, testing progress and record the final results.
By default, wandb is turned off, u could change `wandb_enable` from each training/testing script to enable wandb.

- Install nvdiffrast package
```bash
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
python3 setup.py install
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
cd <path_to_privatar>
mkdir dataset
python3 ./multiface/download_dataset.py --dest "<path_to_Privatar>/dataset" --download_config "./mini_download_config.json"
```
If u follow above instructions, then <path_to_privatar> should be `/work`.

## Step 4: Download pretrained model
The pretrained weights for different users in the provided datasets are collected at this [facial_pretrained_datasets](https://github.com/facebookresearch/multiface/blob/main/documentation/INSTALLATION.md). We use 6795937 base model as the evaluation target. Other structure would work as well. The link for the pretrained model weights of 6795937 is [6795937_base](https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/PretrainedModel/6795937--GHS-base_nosl/best_model.pth)

```bash
cd <path_to_privatar>
mkdir pretrain_model
wget https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/PretrainedModel/6795937--GHS-base_nosl/best_model.pth -O 6795937_best_model.pth
```

# Ready to run?
## Step 1: Training
- Original: multiface baseline using the pretrained model weights
```bash
cd <path_to_privatar>/multiface
python3 launch_train_job_serial_GH200.py
```

- Design Choice 1: directly split mesh and unwrapped texture into two separate path (outsource entire unwrapped texture)
```bash
cd <path_to_privatar>/multiface_direct_split
python3 launch_train_job_serial_GH200.py
```

- Design Choice 2: quantize model to be low precision
```bash
cd <path_to_privatar>/multiface_quantization
python3 launch_train_job_serial_GH200.py
```

- Design Choice 3: prune channels from the decoder to reduce the local computation
```bash
cd <path_to_privatar>/multiface_sparse
python3 launch_train_job_serial_GH200.py
```

- Design Choice 4: decompose "unwrapped texture" into 16 frequency components, but keep all of them run local.
This requires training to be completed.
```bash
cd <path_to_privatar>/multiface_partition_bdct4x4_ibdct
python3 launch_train_job_serial_GH200.py
```

- Design Choice 5: decompose "unwrapped texture" into 16 frequency components, configurable components outsourcing. The number of frequency components are controlled by ```num_freq_comp_outsourced```.
This requires training to be completed.
```bash
cd <path_to_privatar>/multiface_partition_bdct4x4_ibdct_hp
python3 launch_train_job_serial_GH200.py
```

Note: all training results locate at the ```<path_to_privatar>/training_results``` folder.



## Step 2: Testing
- Original: multiface baseline using the pretrained model weights
```bash
cd <path_to_privatar>/multiface
python3 launch_test_job_serial_GH200.py
```

- Design Choice 1: directly split mesh and unwrapped texture into two separate path (outsource entire unwrapped texture)
```bash
cd <path_to_privatar>/multiface_direct_split
python3 launch_test_job_serial_GH200.py
```

- Design Choice 2: quantize model to be low precision
```bash
cd <path_to_privatar>/multiface_quantization
python3 launch_test_job_serial_GH200.py
```

- Design Choice 3: prune channels from the decoder to reduce the local computation
```bash
cd <path_to_privatar>/multiface_sparse
python3 launch_test_job_serial_GH200.py
```

- Design Choice 4: decompose "unwrapped texture" into 16 frequency components, but keep all of them run local.
This requires training to be completed.
```bash
cd <path_to_privatar>/multiface_partition_bdct4x4_ibdct
python3 launch_test_job_serial_GH200.py
```

- Design Choice 5: decompose "unwrapped texture" into 16 frequency components, configurable components outsourcing. The number of frequency components are controlled by ```num_freq_comp_outsourced```.
This requires training to be completed.
```bash
cd <path_to_privatar>/multiface_partition_bdct4x4_ibdct_hp
python3 launch_test_job_serial_GH200.py
```

Note: all testing results locate at the ```<path_to_privatar>/testing_results``` folder.

## Step 3: Noise Calculation
Before calculating noise, two following steps need to be completed.
1. **Complete Training**: Train the model to generate the final weights required for testing.

2. **Perform Testing**: Use the trained model weights to evaluate. 
During testing, the latent codes are stored in the following directory:
   - `./testing_results/<project_name>/latent_code/`
     - `z_<id>.pth`: Latent code for the local path.
     - `z_outsource_<id>.pth`: Latent code for the outsourced path (runs on the cloud). 
Note: the `save_latent_code` command in the script `launch_test_job_serial_GH200.py` controls 
whether to explicitlys store latent code to the external drive.

### Calculate Differential Privacy based Noise
Differential Privacy (DP) noise calculation follows [paper](https://arxiv.org/pdf/1702.07476).
DP requires the prior knowledge of the L2 norm of all outsourced latent codes.
Detailed procedures are written as the comments in the code.

```bash
python3 <path_to_privatar>/Privatar/experiment_scripts/dp_analysis/dp_noise_generation.py
```
Detailed procedures in calculating DP noise are detailed in the function `dp_noise_calculation`.

After this script, the generated noise will show up in the path 
`<path_to_privatar>/experiment_scripts/dp_analysis/generated_dp_noise/dp_noise_ibdct_decoder_<num_outsourced_freq_components>_<mutual_information_bound>.npy`

### Calculate PAC noise
Noise calculation following PAC privacy requires the prior knowledge of the L2 norm of **the covariance** of all outsourced latent codes.
So that PAC privacy could leverage the dimensional differences to generate non-uniform noise for minimizing the overall noise intensity.

```bash
python3 <path_to_privatar>/Privatar/experiment_scripts/pac_analysis/pac_noise_covariance_calculation_GH200.py
```
After this script, the generated noise will show up in the path 
`<path_to_privatar>/experiment_scripts/pac_analysis/noise_covariance/pac_noise_outsourced_ibdct_decoder_<num_outsourced_freq_components>_<mutual_information_bound>.npy`


## Step 4: Noisy Inference
After noise calculation, now we could start testing the actual accuracy and latency 
of noisy inference! Specifically, in horizontally partitioned avatar reconstruction 
flow, generated noise is only injected to the outsourced latent codes.

Noisy inference
```bash
cd <path_to_privatar>/multiface_partition_bdct4x4_ibdct_hp
python3 launch_noisy_test_job_serial_GH200.py
```
This generates detailed accuracy of noisy horizontal partitioned avatar reconstruction 
under various partition configurations.

## Step 5: Empirical Attack
The attacker guesses the expression with the minimal difference through comparing the estimated frequency component to the reference expression, as shown by the Fig. 8 of the paper. Therefore, we need to first define the frequency components of reference expressions.

We provide three different ways of generating reference components, under all of which the attacker shows similar empirical successful rate.

- Method 1: ```accumulate_channel``` mode, where only high frequency components are used as reference. 
- Method 2: ```attack_from_high_frequency_channel``` mode, where only high frequency components are used as reference. 
- Method 3: when both modes are set as False, all 16 frequency components are sorted based on the amibuigity, and the frequency components coming with similar amount of ambuiguity will be merged as one ```if model.normalize_list[freq_pair[0]] + model.normalize_list[freq_pair[1]] < 2:```, where 2 is a random choice that could be changed into different values for obtaining different merging strategies.

Specify the configuration of ```accumulate_channel``` and ```attack_from_high_frequency_channel```, then run the following script to launch attack.

```bash
python3 launch_empirical_attack_GH200.py
```

The final accuracy will be printed out as ```attack_accuracy_mean <final_PSR>```

Note that: ```using_pac_noise``` is the command to control using PAC privacy based noise or Differential Privacy based noise.


# Additional Scripts
We also offer few scripts for researchers, who are interested in exploring the model more deeply, to see different statistics and configurations.

## Script 1: BDCT reconstruction script
```<path_to_privatar>/experiment_scripts/bdct_reconstruction/bdct_4x4_reconstruction_dataloader.ipynb``` includes the script to decomposes given unwrapped texture into different frequency components, and list few different configurations of decomposition.

For each configuration, all frequency components will be written to the folder ```<path_to_privatar>/experiment_scripts/bdct_reconstruction``` for visualization.

## Script 2: Analyze the covariance of each frequency component for different datasets
To help understand the covariance of different frequency components under differnet datasets, we provide the script to perform running profiling of all decomposed frequency components under designated datasets. 

To run it,
```bash
cd <path_to_privatar>/multiface_partition_bdct4x4_ibdct
python3 launch_l2norm_freq_cov_analysis_GH200.py
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

Specifically, the set of images the model would render is detailed ```/scratch2/jianming/work/Privatar_prj/experiment_scripts/render_scripts/test_image_path```.

To launch the image rendering,
```bash
cd <path_to_privatar>/<path_to_configuration>/
python3 launch_test_all_expressions_RTX3090.py
```
where ```<path_to_configuration>``` could be multiface, multiface_partition_bdct4x4_ibdct, multiface_quantization, multiface_direct_split, multiface_partition_bdct4x4_ibdct_hp and multiface_sparse.

It will shows results into the folder ```<path_to_privatar>/render_results/<configuration_name>```


Have Fun! Enjoy! :D
