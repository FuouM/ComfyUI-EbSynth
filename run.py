"""
@author: Fuou Marinas
@title: ComfyUI-EbSynth
@nickname: EbSynth
@description: Run EbSynth in ComfyUI.
"""

import cv2
import numpy as np
import torch

# from ezsynth import Imagesynth
# from ezsynth.utils.guides.guides import GuideFactory
# from .appendix import SequenceManager
from .utils import cv2_to_tensor, tensor_to_cv2, video_tensor_to_cv2

from .Ezsynth.ezsynth.main_ez import ImageSynthBase
from .Ezsynth.ezsynth.aux_classes import RunConfig


class ES_Translate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "source_segment": ("IMAGE",),
                "target_segment": ("IMAGE",),
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
        source_image: torch.Tensor,
        source_segment: torch.Tensor,
        target_segment: torch.Tensor,
        weight: float,
        uniformity: float,
        patch_size: int,
        pyramid_levels: int,
        search_vote_iters: int,
        patch_match_iters: int,
        extra_pass_3x3: bool,
    ):
        print(f"{source_image.shape=}")
        print(f"{source_segment.shape=}")
        print(f"{target_segment.shape=}")

        style_img = tensor_to_cv2(source_image)
        src_img = tensor_to_cv2(source_segment)
        tgt_img = tensor_to_cv2(target_segment)

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
        result, error = ezsynner.run()
        
        print(f"{result.shape=}")
        
        result_tensor = cv2_to_tensor(result)
        error_tensor = cv2_to_tensor(error)
        
        print(f"{result_tensor.shape=}")
        
        return (
            result_tensor,
            error_tensor,
        )


edge_methods = ["PAGE", "PST", "Classic"]  # edge detection
flow_methods = ["RAFT", "DeepFlow"]  # optical flow computation
model_names = ["sintel", "kitti", "chairs"]  # optical flow

default_edge_method = "PAGE"
default_flow_method = "RAFT"
default_model = "sintel"


# class ES_Video:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "source_images": ("IMAGE",),
#                 "style_image": ("IMAGE",),
#                 "style_frame_num": ("INT", {"default": 0, "min": 0, "step": 1}),
#                 "edge_method": (edge_methods, {"default": default_edge_method}),
#                 "flow_method": (flow_methods, {"default": default_flow_method}),
#                 "model_name": (model_names, {"default": default_model}),
#                 "weight": (
#                     "FLOAT",
#                     {"default": 0.9, "min": 0.1},
#                 ),
#                 "uniformity": (
#                     "FLOAT",
#                     {"default": 3500.0, "min": 500.0, "max": 15000.0},
#                 ),
#                 "patch_size": ("INT", {"default": 5, "min": 3, "step": 2}),
#                 "pyramid_levels": ("INT", {"default": 6, "min": 1, "step": 1}),
#                 "search_vote_iters": ("INT", {"default": 12, "min": 1, "step": 1}),
#                 "patch_match_iters": ("INT", {"default": 6, "min": 1, "step": 1}),
#                 "extra_pass_3x3": ("BOOLEAN", {"default": True}),
#             },
#         }

#     RETURN_TYPES = ("IMAGE",)
#     RETURN_NAMES = ("images",)
#     FUNCTION = "todo"
#     CATEGORY = "EbSynth"

#     def todo(
#         self,
#         source_images: torch.Tensor,
#         style_image: torch.Tensor,
#         style_frame_num: int,
#         edge_method: str,
#         flow_method: str,
#         model_name: str,
#         weight: float,
#         uniformity: float,
#         patch_size: int,
#         pyramid_levels: int,
#         search_vote_iters: int,
#         patch_match_iters: int,
#         extra_pass_3x3: bool,
#     ):
#         print(f"{source_images.shape=}")
#         style_img = tensor_to_cv2(style_image)
#         img_sequence = video_tensor_to_cv2(source_images)
#         total_frames = len(img_sequence)
#         if style_frame_num >= total_frames:
#             style_frame_num = total_frames - 1  # consider as last frame

#         guider = GuideFactory(
#             imgsequence=img_sequence,
#             edge_method=edge_method,
#             flow_method=flow_method,
#             model_name=model_name,
#         )
#         manager = SequenceManager(
#             begFrame=0,
#             endFrame=total_frames - 1,
#             style_indexes=[style_frame_num],
#             imgindexes=list(range(total_frames)),
#         )
#         gen_subseqs = manager.run()
#         all_guides = guider.create_all_guides()

#         flow_fwd = all_guides["flow_fwd"]
#         flow_bwd = all_guides["flow_rev"]
#         edge_maps = all_guides["edge"]
#         positional_fwd = all_guides["positional_fwd"]
#         positional_bwd = all_guides["positional_rev"]


#         setup = None

#         pass
