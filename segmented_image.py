from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2


@dataclass
class SegmentedImage:
    img: np.ndarray
    mask: np.ndarray
    score_mask: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.mask.shape[:2] != self.img.shape[:2]:
            raise ValueError("Mask and image dimensions must match")

        if self.score_mask is not None:
            if self.score_mask.shape[:2] != self.img.shape[:2]:
                raise ValueError("Score mask and image dimensions must match")

    @classmethod
    def from_image(cls, img: np.ndarray):
        h, w = img.shape[:2]
        return cls(
            img=img,
            mask=np.zeros((h, w), np.uint8),
            score_mask=np.zeros((h, w), np.float32),
        )

    def copy_with(self, img=None, mask=None, score_mask=None):
        return SegmentedImage(
            img if img is not None else self.img,
            mask if mask is not None else self.mask,
            score_mask if score_mask is not None else self.score_mask,
        )