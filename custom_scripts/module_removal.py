import torch
from collections import OrderedDict

def remove_module_prefix(state_dict):
    """
    Removes the 'module.' prefix from the keys of the state_dict.

    Parameters:
        state_dict (OrderedDict): The state dictionary of the model.

    Returns:
        OrderedDict: A new state dictionary with the 'module.' prefix removed from the keys.
    """
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        # Remove the 'module.' prefix if it exists
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

# Example usage:
if __name__ == "__main__":
    # Load the pre-trained model state dictionary
    pretrained_model_path = '/workspace/uwing2/multiface/pretrained_model/6795937_best_base_model.pth'
    state_dict = torch.load(pretrained_model_path)

    # Remove 'module.' prefix
    cleaned_state_dict = remove_module_prefix(state_dict)
    print(cleaned_state_dict.keys())