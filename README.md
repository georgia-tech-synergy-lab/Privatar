# Privatar: Enabling Privacy-Preserving Real-Time Multi-Users VR Through Secure Outsourcing
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)


# What's Privatar?
Privatar is the first that leverages both local and untrusted cloud to concurrently achieve privacy-preserving multi-user avatar reconstruction.

Functionality-wise, FEATHER supports horizontal partitioning reordering to ensure arbitrary layout changes.

<img src="./figure/arbitrary_reordering_function.png" width="400">

Performance-wise, FEATHER implements Reorder In Reduction (RIR) to hide the reordering latency behind the critical path.


# Structure of the Repo: 
- multiface_partition: Adopt BDCT filter with block=8, which gives 64 different frequency blocks, this does not have having input upsampling and remove three convolution layers.
- multiface_partition_bdct4x4: Decompose unwrapped texture into multiple frequency components -- Adopt BDCT filter with block=4, which gives 16 different frequency blocks, this remove 2 convolution layers, no outsourcing.
- multiface_partition_bdct4x4_hp: Decompose unwrapped texture into multiple frequency components AND horizontally partition all components into local and cloud. Adopt BDCT filter with block=4, which gives 16 different frequency blocks, this remove 2 convolution layers. The number of outsourced frequency components is controlled via ```num_freq_comp_outsourced```.
- multiface_sparse: add sparsity to only decoder of the original VAE model.
- multiface_quantization: change the bitprecision of data into 8-/16-/32-bit integer for the decoder only.


# Installation
We list two flows to install, run and test Privatar because different environments have different package in support. E.g. RTX 3090 supports ```RasterizeGLContext``` but GH200 only supports ```RasterizeCudaContext```. To reduce the effort for setting up configurations for different platforms, we directly prepare two sets of scripts, separately. 

## Step 1: Setup GPU Docker
-  For typical desktop-class GPU such as RTX 3090, we recommend using the conda virtual environment.
```
conda create -n <your_favoriate_name> python=3.7
conda activate <your_favoriate_name>
```

- For cloud-class GPU such as GH200 
We recommand using NVIDIA built-in docker.

Command 1: Download the docker.
If you have Docker 19.03 or later, a typical command to launch the container is:
```
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:xx.xx-py3
```
Note: our experiments base on nvcr.io/nvidia/pytorch:24.01-py3

Command 2: Launch the docker.
```
docker run --gpus all -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --memory 51200m  --rm <docker_name>
```
where <docker_name> refers to the name of your downloaded docker "nvcr.io/nvidia/pytorch:xx.xx-py3"

## Step 2: Install Required Dependency

- Install OS-level dependencies
```bash
$ sudo apt-get install mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev
$ sudo apt-get install mesa-utils
$ glxinfo | grep -i opengl
```

- Install python packages
```
pip3 install torch==1.13.0 -f https://download.pytorch.org/whl/cu121/torch_stable.html
pip3 install Pillow ninja imageio imageio_ffmpeg six tensorboard opencv-python==4.8.0.74 wandb torchjpeg
```
Note: we use "wandb" to track the training, testing progress and record the final results.
By default, wandb is turned off, u could change `wandb_enable` from each training/testing script to enable wandb.

- Download and install NVDiffrast
```
cd <path_to_privatar>
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
python3 setup.py install
```

- Download and install torchjpeg
```
git clone https://github.com/Queuecumber/torchjpeg
cd torchjpeg
pip3 setup.py install
```
We don't recommend installing torchjpeg using pip3 install because it might break other dependencies.

## Step 3: Download Datasets
```
cd <path_to_privatar>
python3 ./multiface/download_dataset.py --dest "<path_to_Privatar>/dataset" --download_config "./mini_download_config.json"
```

## Step 4: Download pretrained model
The pretrained weights for different users in the provided datasets are collected at this [facial_pretrained_datasets](https://github.com/facebookresearch/multiface/blob/main/documentation/INSTALLATION.md). We use 6795937 base model as the evaluation target. Other structure would work as well. The link for the pretrained model weights of 6795937 is [6795937_base](https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/PretrainedModel/6795937--GHS-base_nosl/best_model.pth)

```
cd <path_to_privatar>
mkdir pretrain_model
wget https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/PretrainedModel/6795937--GHS-base_nosl/best_model.pth -O 6795937_best_model.pth
```

# Ready to run?
## Step 1: Training
- Original: multiface baseline using the pretrained model weights
```
cd <path_to_privatar>/multiface
python3 launch_train_job_serial_GH200.py
```

- Design Choice 1: directly split mesh and unwrapped texture into two separate path (outsource entire unwrapped texture)
```
cd <path_to_privatar>/multiface_direct_split
python3 launch_train_job_serial_GH200.py
```

- Design Choice 2: quantize model to be low precision
```
cd <path_to_privatar>/multiface_quantization
python3 launch_train_job_serial_GH200.py
```

- Design Choice 3: prune channels from the decoder to reduce the local computation
```
cd <path_to_privatar>/multiface_sparse
python3 launch_train_job_serial_GH200.py
```

- Design Choice 4: decompose "unwrapped texture" into 16 frequency components, but keep all of them run local.
This requires training to be completed.
```
cd <path_to_privatar>/multiface_partition_bdct4x4_ibdct
python3 launch_train_job_serial_GH200.py
```

- Design Choice 5: decompose "unwrapped texture" into 16 frequency components, configurable components outsourcing. The number of frequency components are controlled by ```num_freq_comp_outsourced```.
This requires training to be completed.
```
cd <path_to_privatar>/multiface_partition_bdct4x4_ibdct_hp
python3 launch_train_job_serial_GH200.py
```

Note: all training results locate at the ```<path_to_privatar>/training_results``` folder.



## Step 2: Testing
- Original: multiface baseline using the pretrained model weights
```
cd <path_to_privatar>/multiface
python3 launch_test_job_serial_GH200.py
```

- Design Choice 1: directly split mesh and unwrapped texture into two separate path (outsource entire unwrapped texture)
```
cd <path_to_privatar>/multiface_direct_split
python3 launch_test_job_serial_GH200.py
```

- Design Choice 2: quantize model to be low precision
```
cd <path_to_privatar>/multiface_quantization
python3 launch_test_job_serial_GH200.py
```

- Design Choice 3: prune channels from the decoder to reduce the local computation
```
cd <path_to_privatar>/multiface_sparse
python3 launch_test_job_serial_GH200.py
```

- Design Choice 4: decompose "unwrapped texture" into 16 frequency components, but keep all of them run local.
This requires training to be completed.
```
cd <path_to_privatar>/multiface_partition_bdct4x4_ibdct
python3 launch_test_job_serial_GH200.py
```

- Design Choice 5: decompose "unwrapped texture" into 16 frequency components, configurable components outsourcing. The number of frequency components are controlled by ```num_freq_comp_outsourced```.
This requires training to be completed.
```
cd <path_to_privatar>/multiface_partition_bdct4x4_ibdct_hp
python3 launch_test_job_serial_GH200.py
```

Note: all testing results locate at the ```<path_to_privatar>/testing_results``` folder.

## Step 3: Noise Calculation
### Calculate Differential Privacy based Noise
Differential Privacy (DP) noise calculation follows [paper]() 
```
python3 <path_to_privatar>/Privatar/custom_scripts/DP_noise_calculation/isotropic_noise_calculation.py
```
It will calculate the L2 norm of the proposed noise 

## Calculate PAC noise

## Step 4: Noisy Inference


### If on custom desktop-level GPU, like RTX 3090

- Original VAE (all local)


- Baseline: Completely Outsource (all outsourced)


- Privatar: Proposed Horizontal Partitioning


### If on custom desktop-level GPU, like GH200

- Original VAE (all local)


- Baseline: Completely Outsource (all outsourced)


- Privatar: Proposed Horizontal Partitioning


