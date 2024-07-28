import cv2
import numpy as np
import torch


def tensor_to_cv2(image: torch.Tensor):
    np_array: np.ndarray = image.squeeze(0).cpu().numpy()

    np_array = (np_array * 255).astype(np.uint8)
    np_array = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)
    return np_array


def to_np(tsr: torch.Tensor, color=cv2.COLOR_RGB2BGR) -> np.ndarray:
    np_arr = tsr.cpu().numpy()
    np_arr = (np_arr * 255).astype(np.uint8)
    np_arr = cv2.cvtColor(np_arr, color)
    return np_arr


def batched_tensor_to_cv2_list(
    tensor_imgs: torch.Tensor, color=cv2.COLOR_RGB2BGR
) -> list[np.ndarray]:
    return [to_np(tsr, color) for tsr in tensor_imgs]


def cv2_img_to_tensor(np_array: np.ndarray):
    tensor = cv2_to_a_tensor(np_array).unsqueeze(0)
    return tensor


def cv2_to_a_tensor(np_arr: np.ndarray):
    np_arr = cv2.cvtColor(np_arr, cv2.COLOR_BGR2RGB)
    np_arr = np_arr.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_arr)
    return tensor


def out_video(predictions: list[np.ndarray]):
    out_tensor_list = []
    for i in predictions:
        out_img = cv2_to_a_tensor(i)
        out_tensor_list.append(out_img)
    images = torch.stack(out_tensor_list, dim=0)
    return images


def process_msk_lst(msks: list[np.ndarray]):
    msk_arr_seq: list[np.ndarray] = []
    for msk_fr in msks:
        _, msk = cv2.threshold(msk_fr, 1, 255, cv2.THRESH_BINARY)
        msk_arr_seq.append(msk)
    return msk_arr_seq
