from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
import cv2

import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from modules.cv_modules.segmentation import RegionGrowing
from modules.cv_modules.segmented_image import SegmentedImage
from kivy.graphics import Canvas, Color, Rectangle, RoundedRectangle
import cv2
import numpy as np

import cv2
import numpy as np

from kivy.uix.widget import Widget
from kivy.graphics import RoundedRectangle
from kivy.graphics.texture import Texture


class SegmentationPicker(Widget):
    points = []
    def __init__(self, image_path: str = None, image = None, **kwargs):
        super().__init__(**kwargs)

        if image_path:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
        self.original_image = image

        self.display_image = self.original_image.copy()
        self.segmenter = RegionGrowing()
        self.points = []

        self.mask = np.zeros(self.original_image.shape, np.int8)

        self.img_h, self.img_w = self.display_image.shape[:2]

        self.texture = Texture.create(
            size=(self.img_w, self.img_h),
            colorfmt="rgb"
        )

        with self.canvas:
            self.rect = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[20],
                texture=self.texture
            )

        self.bind(pos=self.update_rect, size=self.update_rect)
        self.render()
    
    def __init__(self, image_path: str = None, image=None, **kwargs):
        super().__init__(**kwargs)

        if image is not None:
            # Direct cv2 image
            self.original_image = image.copy()

        elif image_path is not None:
            # Load from file
            self.original_image = cv2.imread(image_path)

            if self.original_image is None:
                raise ValueError(f"Could not load image: {image_path}")

        else:
            raise ValueError("Either image_path or image must be provided")

        self.display_image = self.original_image.copy()

        self.segmenter = RegionGrowing()
        self.selected_points = []

        self.img_h, self.img_w = self.display_image.shape[:2]

        self.texture = Texture.create(
            size=(self.img_w, self.img_h),
            colorfmt="rgb"
        )

        with self.canvas:
            self.rect = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[20],
                texture=self.texture
            )

        self.bind(pos=self.render, size=self.render)

        self.render()

    def update_rect(self, *args):
        if self.width <= 0 or self.height <= 0:
            return

        img_aspect = self.img_w / self.img_h
        widget_aspect = self.width / self.height

        if widget_aspect > img_aspect:
            # Limited by height
            display_h = self.height
            display_w = display_h * img_aspect
        else:
            # Limited by width
            display_w = self.width
            display_h = display_w / img_aspect

        display_x = self.x + (self.width - display_w) / 2
        display_y = self.y + (self.height - display_h) / 2

        self.rect.pos = (display_x, display_y)
        self.rect.size = (display_w, display_h)

    def render(self, *args):
        rgb = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)

        self.texture.blit_buffer(
            memoryview(rgb.ravel()),
            colorfmt="rgb",
            bufferfmt="ubyte"
        )

        self.update_rect()
        self.canvas.ask_update()

    def widget_to_image_coords(self, x, y):
        rx, ry = self.rect.pos
        rw, rh = self.rect.size

        # Ignore clicks outside the displayed image rect
        if not (rx <= x <= rx + rw and ry <= y <= ry + rh):
            return None

        rel_x = (x - rx) / rw
        rel_y = (y - ry) / rh

        img_x = int(rel_x * (self.img_w - 1))
        img_y = int((1.0 - rel_y) * (self.img_h - 1))

        return img_x, img_y

    def draw_points(self):
        self.display_image = self.original_image.copy()

        for px, py in self.points:
            cv2.circle(self.display_image, (px, py), 5, (0, 0, 255), -1)

    def run_segmentation(self):
        if not self.points:
            return

        self.segmenter.clear_seeds()
        for px, py in self.points:
            self.segmenter.add_seed(px, py)

        mask, contour = self.segmenter.segment(self.original_image)
        self.mask = mask


        overlay = self.segmenter.create_overlay(self.original_image, mask, alpha=0.5, color=(0, 255, 0))

        for px, py in self.points:
            cv2.circle(overlay, (int(px), int(py)), 5, (0, 0, 255), -1)

        self.display_image = overlay

    def on_touch_down(self, touch):
        coords = self.widget_to_image_coords(touch.x, touch.y)
        if coords is None:
            return super().on_touch_down(touch)

        self.points.append(coords)

        print(self.points)

        self.run_segmentation()
        self.render()
        return True
    
    def get_segmented_result(self) -> SegmentedImage:
        if self.mask is None:
            h, w = self.original_image.shape[:2]
            mask_to_return = np.zeros((h, w), dtype=np.uint8)
        else:
            mask_to_return = self.mask

        # Cria e retorna a instância do objeto
        return SegmentedImage(self.original_image, mask_to_return)
