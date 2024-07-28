"""
@author: Fuou Marinas
@title: ComfyUI-EbSynth
@nickname: EbSynth
@description: Run EbSynth in ComfyUI.
"""

import cv2
import torch

from .constants import EBTYPES, MAX_GUIDES, ONLY_DEFAULT_MODE, ONLY_MODES
from .Ezsynth.ezsynth.aux_classes import RunConfig
from .Ezsynth.ezsynth.constants import (
    DEFAULT_EDGE_METHOD,
    DEFAULT_FLOW_MODEL,
    EDGE_METHODS,
    FLOW_MODELS,
)
from .Ezsynth.ezsynth.main_ez import EzsynthBase, ImageSynthBase
from .utils import (
    batched_tensor_to_cv2_list,
    cv2_img_to_tensor,
    out_video,
    process_msk_lst,
)


def create_guide_arg(i: int):
    return {
        f"src_img_{i}": ("IMAGE",),
        f"tgt_img_{i}": ("IMAGE",),
        f"weight_{i}": (
            "FLOAT",
            {"default": 1.0, "min": 0.1},
        ),
    }


def build_guide_args(max_guides=MAX_GUIDES):
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

        for i in range(MAX_GUIDES):
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
        required = {
            "style_image": ("IMAGE",),
            "source_image": ("IMAGE",),
            "target_image": ("IMAGE",),
            "weight": (
                "FLOAT",
                {"default": 0.9, "min": 0.1},
            ),
        }
        required.update(EBTYPES)
        return {
            "required": required,
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

        style_img = batched_tensor_to_cv2_list(style_image)[0]
        src_img = batched_tensor_to_cv2_list(source_image)[0]
        tgt_img = batched_tensor_to_cv2_list(target_image)[0]

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

        result_tensor = cv2_img_to_tensor(result)
        error_tensor = cv2_img_to_tensor(error)

        print(f"{result_tensor.shape=}")

        return (
            result_tensor,
            error_tensor,
        )


class ES_VideoTransfer:
    @classmethod
    def INPUT_TYPES(s):
        required = {
            "source_video": ("IMAGE",),
            "style_images": ("IMAGE",),
            "style_idxes": ("STRING", {"default": "0"}),
            "edge_method": (EDGE_METHODS, {"default": DEFAULT_EDGE_METHOD}),
            "flow_model": (FLOW_MODELS, {"default": DEFAULT_FLOW_MODEL}),
            "only_mode": (ONLY_MODES, {"default": ONLY_DEFAULT_MODE}),
            "do_mask": (
                "BOOLEAN",
                {"default": False},
            ),
            "pre_mask": (
                "BOOLEAN",
                {"default": False},
            ),
            "feather": (
                "INT",
                {"default": 5, "min": 0, "step": 1},
            ),
            "style_weight": (
                "FLOAT",
                {"default": 6.0, "min": 0.1},
            ),
            "edge_weight": (
                "FLOAT",
                {"default": 1.0, "min": 0.1},
            ),
            "warp_weight": (
                "FLOAT",
                {"default": 0.5, "min": 0.1},
            ),
            "pos_weight": (
                "FLOAT",
                {"default": 2.0, "min": 0.1},
            ),
        }
        required.update(EBTYPES)
        required.update(
            {
                "use_gpu_hist_blend": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "use_lsqr": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "use_poisson_cupy": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "poisson_maxiter": (
                    "INT",
                    {"default": 0, "min": 0, "step": 1},
                ),
            }
        )
        return {
            "required": required,
            "optional": {
                "source_mask": ("IMAGE",),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "result_video",
        "error_video",
    )
    FUNCTION = "todo"
    CATEGORY = "EbSynth"

    def todo(
        self,
        source_video: torch.Tensor,
        style_images: torch.Tensor,
        style_idxes: str,
        edge_method: str,
        flow_model: str,
        only_mode: str,
        # Masking params
        do_mask: bool,
        pre_mask: bool,
        feather: float,
        # Ebsynth guide weights params
        style_weight: float,
        edge_weight: float,
        warp_weight: float,
        pos_weight: float,
        # Ebsynth gen params
        uniformity: float,
        patch_size: int,
        pyramid_levels: int,
        search_vote_iters: int,
        patch_match_iters: int,
        extra_pass_3x3: bool,
        # Blending params
        use_gpu_hist_blend: bool,
        use_lsqr: bool,
        use_poisson_cupy: bool,
        poisson_maxiter: int,
        source_mask: torch.Tensor | None = None,
    ):
        print(f"{source_video.shape=}")
        print(f"{style_images.shape=}")

        img_frs_seq = batched_tensor_to_cv2_list(source_video)
        stl_frs = batched_tensor_to_cv2_list(style_images)
        stl_idxes = sorted(deserialize_integers(style_idxes))
        
        if len(stl_frs) != len(stl_idxes):
            raise ValueError(f"Style indices mismatch: There are {len(stl_frs)}, but only [{stl_idxes}]")
        
        if source_mask is not None:
            print(f"{source_mask.shape=}")
            msk_frs_seq = process_msk_lst(
                batched_tensor_to_cv2_list(source_mask, color=cv2.COLOR_RGB2GRAY)
            )
        else:
            msk_frs_seq = None

        ezrunner = EzsynthBase(
            style_frs=stl_frs,
            style_idxes=stl_idxes,
            img_frs_seq=img_frs_seq,
            cfg=RunConfig(
                only_mode=only_mode,
                pre_mask=pre_mask,
                feather=feather,
                uniformity=uniformity,
                patchsize=patch_size,
                pyramidlevels=pyramid_levels,
                searchvoteiters=search_vote_iters,
                patchmatchiters=patch_match_iters,
                extrapass3x3=extra_pass_3x3,
                img_wgt=style_weight,
                edg_wgt=edge_weight,
                wrp_wgt=warp_weight,
                pos_wgt=pos_weight,
                use_gpu=use_gpu_hist_blend,
                use_lsqr=use_lsqr,
                use_poisson_cupy=use_poisson_cupy,
                poisson_maxiter=poisson_maxiter if poisson_maxiter > 0 else None,
            ),
            edge_method=edge_method,
            raft_flow_model_name=flow_model,
            do_mask=do_mask,
            msk_frs_seq=msk_frs_seq,
        )

        stylized_frames, err_frames = ezrunner.run_sequences()
        style_tensor = out_video(stylized_frames)
        err_tensor = out_video(err_frames)

        return (
            style_tensor,
            err_tensor,
        )


def serialize_floats(lst: list[float]):
    return ",".join(map(str, lst))


def deserialize_floats(floats_str: str):
    return [float(w) for w in floats_str.split(",") if w.strip()]


def serialize_integers(int_list):
    return "_".join(map(str, int_list))


def deserialize_integers(int_string):
    return list(map(int, int_string.split("_")))
