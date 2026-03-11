import cv2
import numpy as np

from typing import Iterable, List, Optional

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button

from mobileapp.my_widgets.segmentation_picker import SegmentationPicker
from modules.cv_modules.propagation import SegmentationPropagator
from modules.cv_modules.segmented_image import SegmentedImage

def picker_to_segmented_image(picker: SegmentationPicker) -> SegmentedImage:
    original = picker.original_image.copy()
    points = list(picker.points)

    if not points:
        raise ValueError("No seed points selected.")

    result = picker.segmenter.segment(original, points)

    if isinstance(result, SegmentedImage):
        return result

    score_mask = None

    if isinstance(result, tuple):
        first = result[0]
        second = result[1] if len(result) > 1 else None

        if isinstance(first, SegmentedImage):
            return first

        if isinstance(first, np.ndarray) and len(first.shape) == 2:
            return SegmentedImage(
                img=original,
                mask=(first > 0).astype(np.uint8),
                score_mask=second if isinstance(second, np.ndarray) else None
            )

        if (
            isinstance(first, np.ndarray)
            and len(first.shape) == 3
            and isinstance(second, np.ndarray)
            and len(second.shape) == 2
        ):
            return SegmentedImage(
                img=first,
                mask=(second > 0).astype(np.uint8),
                score_mask=None
            )

        result = first

    if isinstance(result, np.ndarray) and len(result.shape) == 2:
        return SegmentedImage(
            img=original,
            mask=(result > 0).astype(np.uint8),
            score_mask=None
        )

    raise TypeError(f"Cannot convert segmentation result of type {type(result)} to SegmentedImage")


def load_video_frames(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames: List[np.ndarray] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"Video contains no readable frames: {video_path}")

    return frames


def _extract_mask_from_segmentation_result(
    result,
    original_image: np.ndarray
) -> np.ndarray:
    """
    Converts the output of RegionGrowing into a binary mask.

    Supported possibilities:
    - np.ndarray mask with shape (H, W)
    - np.ndarray BGR image with same shape as original
    - SegmentedImage
    """
    if isinstance(result, SegmentedImage):
        return result.mask.astype(np.uint8)

    if not isinstance(result, np.ndarray):
        raise TypeError(
            "RegionGrowing returned an unsupported type. "
            "Expected np.ndarray or SegmentedImage."
        )

    # Case 1: already a single-channel mask
    if len(result.shape) == 2:
        return (result > 0).astype(np.uint8)

    # Case 2: returned an image instead of a mask
    # Infer the mask by comparing against the original image
    if result.shape == original_image.shape:
        diff = np.any(result != original_image, axis=2)
        return diff.astype(np.uint8)

    raise ValueError(
        "Could not infer mask from RegionGrowing output. "
        f"Got shape {result.shape}, expected either "
        f"(H, W) or {original_image.shape}."
    )


def picker_to_segmented_image(picker: SegmentationPicker) -> SegmentedImage:
    """
    Builds a SegmentedImage from the current state of SegmentationPicker
    without modifying SegmentationPicker itself.
    """
    if not hasattr(picker, "original_image"):
        raise AttributeError("SegmentationPicker must expose 'original_image'.")

    if not hasattr(picker, "selected_points"):
        raise AttributeError("SegmentationPicker must expose 'selected_points'.")

    if not hasattr(picker, "segmenter"):
        raise AttributeError("SegmentationPicker must expose 'segmenter'.")

    original = picker.original_image.copy()
    points = list(picker.selected_points)

    if not points:
        raise ValueError("No seed points were selected in the SegmentationPicker.")

    result = picker.segmenter.segment(original, points)
    mask = _extract_mask_from_segmentation_result(result, original)

    return SegmentedImage(
        img=original,
        mask=mask,
        score_mask=None
    )


class VideoSegmentationRoot(BoxLayout):
    def __init__(self, video_path: str = "input.mp4", **kwargs):
        super().__init__(orientation="vertical", **kwargs)

        self.video_path = video_path
        self.frames = load_video_frames(video_path)

        self.base_frame = self.frames[0]
        self.remaining_frames = self.frames[1:]

        self.propagator = SegmentationPropagator()
        self.results: Optional[List[SegmentedImage]] = None

        self.picker = SegmentationPicker(image=self.base_frame)

        self.run_button = Button(
            text="Propagate segmentation through video",
            size_hint=(1, None),
            height=60
        )
        self.run_button.bind(on_press=self.on_run_pressed)

        self.add_widget(self.picker)
        self.add_widget(self.run_button)

    def on_run_pressed(self, *args):
        base_segmented = picker_to_segmented_image(self.picker)

        self.results = self.propagator.propagate_video(
            base_frame=base_segmented,
            frames=self.remaining_frames
        )

        print(f"Propagation finished. Generated {len(self.results)} segmented frames.")


class VideoSegmentationApp(App):
    def build(self):
        return VideoSegmentationRoot(video_path="input.mp4")
    
if __name__ == "__main__":
    VideoSegmentationApp().run()