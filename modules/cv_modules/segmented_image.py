from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2


@dataclass
class SegmentedImage:
    img: np.ndarray
    mask: Optional[np.ndarray] = None
    score_mask: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.mask is not None:
            if self.mask.shape[:2] != self.img.shape[:2]:
                raise ValueError("Mask and image dimensions must match")
        else:
            self.mask = np.zeros(self.img.shape[:2], np.int8)

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
    
    def get_masked_image(self, alpha=0.5, color=(0, 255, 0)):
        if self.mask is None:
            return self.image.copy()

        if len(self.img.shape) == 2:
            base_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        else:
            base_img = self.img.copy()

        color_layer = np.zeros_like(base_img)
        color_layer[self.mask > 0] = color

        result = cv2.addWeighted(base_img, 1.0, color_layer, alpha, 0)
        
        return result