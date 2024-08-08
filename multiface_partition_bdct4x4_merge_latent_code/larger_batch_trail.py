from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from dataset import Dataset
import torch
import json

torch.distributed.init_process_group(backend="nccl")

local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

f = open("/workspace/uwing2/Privatar/multiface_partition/camera_configs/camera-split-config_6795937.json")
camera_config = json.load(f)

camera_config = camera_config["full"]

dataset_train = Dataset(
    "/workspace/uwing2/multiface/dataset/m--20180227--0000--6795937--GHS",
    "/workspace/uwing2/multiface/dataset/m--20180227--0000--6795937--GHS/KRT",
    # "/workspace/uwing2/multiface/dataset/m--20180227--0000--6795937--GHS/frame_list.txt",
    "/workspace/uwing2/multiface/reduced_frame_list.txt",
    1024,
    camset=None if camera_config is None else camera_config["train"],
    exclude_prefix=["EXP_ROM", "EXP_free_face"],
)

train_sampler = DistributedSampler(dataset_train)
train_random_sampler = RandomSampler(dataset_train)

# train_loader1 = DataLoader(
#     dataset_train,
#     33,
#     sampler=train_sampler,
#     num_workers=0,
# )
# print(f"total number of iteration, batch=33, num_worker=0 -- {len(train_loader1)}")
# for i, data in enumerate(train_loader1):
#     if(i == 0):
#         print(i, data['tex'].shape)


train_loader2 = DataLoader(
    dataset_train,
    16,
    sampler=train_sampler,
    num_workers=8,
)
print(f"total number of iteration, batch=16, num_worker=8 -- {len(train_loader2)}")
for i, data in enumerate(train_loader2):
    print(i, data['tex'].shape)


# train_loader3 = DataLoader(
#     dataset_train,
#     33,
#     sampler=train_random_sampler,
#     num_workers=0,
# )
# print(f"total number of iteration under random sampler: batch=33, num_worker=0 -- {len(train_loader3)}")
# for i, data in enumerate(train_loader3):
#     if(i == 0):
#         print(i, data['tex'].shape)
    
