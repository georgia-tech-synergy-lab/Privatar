import torch
torch.cuda.empty_cache()           # releases **unused** cached blocks back to the driver
torch.cuda.ipc_collect()           # releases inter-process handles
torch.cuda.reset_peak_memory_stats()