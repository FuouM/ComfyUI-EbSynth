MAX_GUIDES = 7
ONLY_MODES = [
    "forward",
    "reverse",
    "none",
]
ONLY_DEFAULT_MODE = "none"

EBTYPES = {
    "uniformity": (
        "FLOAT",
        {"default": 3500.0, "min": 500.0, "max": 15000.0},
    ),
    "patch_size": ("INT", {"default": 5, "min": 3, "step": 2}),
    "pyramid_levels": ("INT", {"default": 6, "min": 1, "step": 1}),
    "search_vote_iters": ("INT", {"default": 12, "min": 1, "step": 1}),
    "patch_match_iters": ("INT", {"default": 6, "min": 1, "step": 1}),
    "extra_pass_3x3": ("BOOLEAN", {"default": True}),
}
