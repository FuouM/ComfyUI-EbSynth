# ComfyUI-EbSynth
EbSynth in ComfyUI

Params:
* uniformity: uniformity weight for the style transfer
* patchsize: odd sizes only, use 5 for 5x5 patch, 7 for 7x7, etc. 
* pyramidlevels = 6  # Larger Values useful for things like color transfer
* searchvoteiters = 12  # how many search/vote iters to perform at each level
* patchmatchiters = 6  # how many Patch-Match iters to perform at each level
* extrapass3x3 = True  # perform additional polishing pass with 3x3 patches at the finest level, use 0 to disable
