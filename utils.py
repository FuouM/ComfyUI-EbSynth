import torch
import numpy as np
import cv2


def tensor_to_cv2(image: torch.Tensor):
    np_array: np.ndarray = image.squeeze(0).cpu().numpy()

    np_array = (np_array * 255).astype(np.uint8)
    np_array = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)
    return np_array


def cv2_to_tensor(np_array: np.ndarray):
    np_array = cv2.cvtColor(np_array, cv2.COLOR_BGR2RGB)
    np_array = np_array.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_array)
    tensor = tensor.unsqueeze(0)
    return tensor

def video_tensor_to_cv2(images: torch.Tensor):
    # Scale to 0-255 range
    images = (images * 255).byte()
    
    # Swap color channels (RGB to BGR)
    images = images[:, [2, 1, 0], :, :]
    
    # Convert to list of numpy arrays
    return [frame.cpu().numpy() for frame in images]