"""Submission-pipeline helper shared by v4 OOF evaluation.

The historical v3 search/inference entry point was removed. The helper below is
unchanged: grayscale -> 512x512 bilinear -> >0 -> C-order flatten.
This is a local submission-pipeline proxy, not a verified server metric.
"""
import numpy as np
from PIL import Image
import torchvision


def official_mask(arr_u8):
    """Replicates segmentation_to_csv.turn_to_str up to the binary mask (before RLE)."""
    im = Image.fromarray(arr_u8).convert("L").resize((512, 512), Image.Resampling.BILINEAR)
    t = torchvision.transforms.ToTensor()(im)
    return (t > 0).numpy().astype(np.uint8).flatten()
