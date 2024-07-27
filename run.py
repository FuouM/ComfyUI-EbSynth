"""
@author: Fuou Marinas
@title: ComfyUI-EbSynth
@nickname: EbSynth
@description: Run EbSynth in ComfyUI.
"""

import cv2
import numpy as np
import torch

from .utils import (
    batched_tensor_to_cv2_list,
    cv2_to_tensor,
    tensor_to_cv2,
    video_tensor_to_cv2,
)

from .Ezsynth.ezsynth.main_ez import ImageSynthBase
from .Ezsynth.ezsynth.aux_classes import RunConfig


def create_guide_arg(i: int):
    return {
        f"src_img_{i}": ("IMAGE",),
        f"tgt_img_{i}": ("IMAGE",),
        f"weight_{i}": (
            "FLOAT",
            {"default": 1.0, "min": 0.1},
        ),
    }


def build_guide_args(max_guides=7):
    required = create_guide_arg(0)
    optional = dict()
    for i in range(1, max_guides):
        optional.update(create_guide_arg(i))
    return {"required": required, "optional": optional}


class ES_Guides7:
    @classmethod
    def INPUT_TYPES(s):
        return build_guide_args()

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "STRING",
    )
    RETURN_NAMES = (
        "src_imgs",
        "tgt_imgs",
        "wgts",
    )

    FUNCTION = "todo"
    CATEGORY = "EbSynth"

    def todo(self, **args):
        src_imgs = []
        tgt_imgs = []
        weights = []

        for i in range(7):
            src_key = f"src_img_{i}"
            tgt_key = f"tgt_img_{i}"
            weight_key = f"weight_{i}"
            if src_key in args and tgt_key in args:
                src_imgs.append(args[src_key])
                tgt_imgs.append(args[tgt_key])
                weights.append(args[weight_key])

        src_tensor = torch.cat(src_imgs, dim=0)
        tgt_tensor = torch.cat(tgt_imgs, dim=0)
        weights_str = serialize_floats(weights)

        return (src_tensor, tgt_tensor, weights_str)


class ES_Translate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "style_image": ("IMAGE",),
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "weight": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.1},
                ),
                "uniformity": (
                    "FLOAT",
                    {"default": 3500.0, "min": 500.0, "max": 15000.0},
                ),
                "patch_size": ("INT", {"default": 5, "min": 3, "step": 2}),
                "pyramid_levels": ("INT", {"default": 6, "min": 1, "step": 1}),
                "search_vote_iters": ("INT", {"default": 12, "min": 1, "step": 1}),
                "patch_match_iters": ("INT", {"default": 6, "min": 1, "step": 1}),
                "extra_pass_3x3": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "src_imgs": ("IMAGE",),
                "tgt_imgs": ("IMAGE",),
                "wgts": ("STRING", {}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "result_img",
        "error_img",
    )
    FUNCTION = "todo"
    CATEGORY = "EbSynth"

    def todo(
        self,
        style_image: torch.Tensor,
        source_image: torch.Tensor,
        target_image: torch.Tensor,
        weight: float,
        uniformity: float,
        patch_size: int,
        pyramid_levels: int,
        search_vote_iters: int,
        patch_match_iters: int,
        extra_pass_3x3: bool,
        src_imgs: torch.Tensor | None = None,
        tgt_imgs: torch.Tensor | None = None,
        wgts: str | None = None,
    ):
        print(f"{style_image.shape=}")
        print(f"{source_image.shape=}")
        print(f"{target_image.shape=}")

        style_img = tensor_to_cv2(style_image)
        src_img = tensor_to_cv2(source_image)
        tgt_img = tensor_to_cv2(target_image)

        guides = []

        if src_imgs is not None and tgt_imgs is not None and wgts is not None:
            g_srcs = batched_tensor_to_cv2_list(src_imgs)
            g_tgt = batched_tensor_to_cv2_list(tgt_imgs)
            g_wgts = deserialize_floats(wgts)
            for g_src, g_tgt, g_wgt in zip(g_srcs, g_tgt, g_wgts):
                guides.append((g_src, g_tgt, g_wgt))

        print(wgts)

        params = {
            "style_img": style_img,
            "src_img": src_img,
            "tgt_img": tgt_img,
            "cfg": RunConfig(
                uniformity=uniformity,
                patchsize=patch_size,
                pyramidlevels=pyramid_levels,
                searchvoteiters=search_vote_iters,
                patchmatchiters=patch_match_iters,
                extrapass3x3=extra_pass_3x3,
                img_wgt=weight,
            ),
        }

        ezsynner = ImageSynthBase(**params)
        result, error = ezsynner.run(guides=guides)

        print(f"{result.shape=}")

        result_tensor = cv2_to_tensor(result)
        error_tensor = cv2_to_tensor(error)

        print(f"{result_tensor.shape=}")

        return (
            result_tensor,
            error_tensor,
        )


def serialize_floats(lst: list[float]):
    return ",".join(map(str, lst))


def deserialize_floats(floats_str: str):
    return [float(w) for w in floats_str.split(",") if w.strip()]
