import os
import cv2
import numpy as np

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, RoundedRectangle
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget

from modules.cv_modules.segmentation import RegionGrowing
from modules.cv_modules.propagation import SegmentationPropagator
from modules.cv_modules.segmented_image import SegmentedImage


class VideoSegmentationWidget(SegmentationPropagator):
    def __init__(self, video_path: str, output_path: str = "propagated_output.mp4", **kwargs):
        super().__init__(**kwargs)

        self.video_path = video_path
        self.output_path = output_path

        self.frames, self.fps = self.load_video(video_path)
        if not self.frames:
            raise ValueError(f"Could not load frames from {video_path}")

        self.first_frame = self.frames[0].copy()
        self.display_image = self.first_frame.copy()

        self.img_h, self.img_w = self.first_frame.shape[:2]

        self.segmenter = RegionGrowing()
        self.propagator = SegmentationPropagator(segmenter=RegionGrowing())

        self.points = []
        self.initial_segmented = None
        self.processing = False

        self.texture = Texture.create(
            size=(self.img_w, self.img_h),
            colorfmt="rgb"
        )

        with self.canvas:
            Color(0.10, 0.10, 0.10, 1.0)
            self.bg_rect = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[18]
            )

            self.image_rect = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[18],
                texture=self.texture
            )

        self.bind(pos=self.update_rect, size=self.update_rect)
        self.render()

    # ---------------------------------------------------------
    # VIDEO LOADING
    # ---------------------------------------------------------

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            fps = 30.0

        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)

        cap.release()
        return frames, fps

    # ---------------------------------------------------------
    # UI / RENDER
    # ---------------------------------------------------------

    def update_rect(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size

        if self.width <= 0 or self.height <= 0:
            return

        img_aspect = self.img_w / self.img_h
        widget_aspect = self.width / self.height

        if widget_aspect > img_aspect:
            display_h = self.height
            display_w = display_h * img_aspect
        else:
            display_w = self.width
            display_h = display_w / img_aspect

        display_x = self.x + (self.width - display_w) / 2
        display_y = self.y + (self.height - display_h) / 2

        self.image_rect.pos = (display_x, display_y)
        self.image_rect.size = (display_w, display_h)

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

        rel_x = (x - self.x) / self.width
        rel_y = (y - self.y) / self.height

        img_x = int(rel_x * (self.img_w - 1))
        img_y = int(rel_y * (self.img_h - 1))

        return img_x, img_y

    # ---------------------------------------------------------
    # SEGMENTATION PREVIEW
    # ---------------------------------------------------------

    def build_initial_segmentation(self):
        if not self.points:
            self.initial_segmented = None
            self.display_image = self.first_frame.copy()
            self.render()
            return

        mask, contours = self.segmenter.segment(self.first_frame)

        score_mask = mask.astype(np.float32)
        self.initial_segmented = SegmentedImage(
            self.first_frame.copy(),
            mask,
            score_mask
        )

        preview = self.overlay_mask(self.first_frame, mask, color=(0, 255, 0), alpha=0.35)
        preview = self.draw_contours(preview, contours, color=(0, 255, 0), thickness=2)

        for px, py in self.points:
            cv2.circle(preview, (px, py), 5, (0, 0, 255), -1)

        self.display_image = preview
        self.render()

    def overlay_mask(self, img, mask, color=(0, 255, 0), alpha=0.35):
        out = img.copy()
        colored = np.zeros_like(img)
        colored[mask > 0] = color

        idx = mask > 0
        out[idx] = cv2.addWeighted(
            img[idx], 1.0 - alpha,
            colored[idx], alpha,
            0
        )
        return out

    def draw_contours(self, img, contours, color=(0, 255, 0), thickness=2):
        out = img.copy()
        if contours:
            cv2.drawContours(out, contours, -1, color, thickness)
        return out

    # ---------------------------------------------------------
    # INPUT
    # ---------------------------------------------------------

    def on_touch_down(self, touch):
        if self.processing:
            return super().on_touch_down(touch)

        # right click clears
        if hasattr(touch, "button") and touch.button == "right":
            self.points.clear()
            self.segmenter.clear_seeds()
            self.initial_segmented = None
            self.display_image = self.first_frame.copy()
            self.render()
            return True

        coords = self.widget_to_image_coords(touch.x, touch.y)
        if coords is None:
            return super().on_touch_down(touch)

        img_x, img_y = coords

        self.points.append((img_x, img_y))
        self.segmenter.add_seed(img_x, img_y)

        self.build_initial_segmentation()
        return True

    def on_key_down(self, window, key, scancode, codepoint, modifiers):
        # Enter
        if key in (13, 271) and not self.processing:
            self.start_propagation()
            return True

        # Backspace undo
        if key == 8 and not self.processing:
            if self.points:
                self.points.pop()
                self.segmenter.undo_last_seed()
                self.build_initial_segmentation()
            return True

        # C clear
        if codepoint and codepoint.lower() == "c" and not self.processing:
            self.points.clear()
            self.segmenter.clear_seeds()
            self.initial_segmented = None
            self.display_image = self.first_frame.copy()
            self.render()
            return True

        return False

    # ---------------------------------------------------------
    # PROPAGATION
    # ---------------------------------------------------------

    def start_propagation(self):
        if self.initial_segmented is None:
            print("Select the object first.")
            return

        self.processing = True
        Clock.schedule_once(lambda dt: self.process_video(), 0)

    def process_video(self):
        writer = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.img_w, self.img_h)
        )

        prev_segmented = self.initial_segmented
        first_annotated = self.annotate_segmented(prev_segmented)
        writer.write(first_annotated)
        self.display_image = first_annotated
        self.render()

        processed_count = 1

        for frame in self.frames[1:]:
            next_segmented = self.propagator.propagate(prev_segmented, frame)

            if self.should_stop(prev_segmented, next_segmented):
                print(f"Stopping propagation at frame {processed_count}.")
                break

            annotated = self.annotate_segmented(next_segmented)
            writer.write(annotated)

            self.display_image = annotated
            self.render()

            prev_segmented = next_segmented
            processed_count += 1

        writer.release()
        self.processing = False

        print(f"Saved: {os.path.abspath(self.output_path)}")
        print(f"Frames written: {processed_count}")

    def annotate_segmented(self, segmented: SegmentedImage):
        mask = segmented.mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        out = self.overlay_mask(segmented.img, mask, color=(0, 255, 0), alpha=0.35)
        out = self.draw_contours(out, contours, color=(0, 255, 0), thickness=2)
        return out

    def should_stop(self, prev_segmented: SegmentedImage, next_segmented: SegmentedImage):
        mask = next_segmented.mask
        area = int(np.count_nonzero(mask))

        if area == 0:
            print("Stop reason: empty mask.")
            return True

        prev_area = int(np.count_nonzero(prev_segmented.mask))
        if prev_area == 0:
            return True

        area_ratio = area / prev_area

        # If area collapses too much or explodes too much, consider it lost
        if area_ratio < 0.15:
            print(f"Stop reason: area collapsed too much ({area_ratio:.3f}).")
            return True

        if area_ratio > 3.5:
            print(f"Stop reason: area exploded too much ({area_ratio:.3f}).")
            return True

        # If score mask exists, require some confidence mass
        if getattr(next_segmented, "score_mask", None) is not None:
            score_mass = float(next_segmented.score_mask[mask > 0].mean()) if np.any(mask > 0) else 0.0
            if score_mass < 80.0:
                print(f"Stop reason: low score mass ({score_mass:.2f}).")
                return True

        return False


class RootUI(BoxLayout):
    def __init__(self, video_path: str, output_path: str, **kwargs):
        super().__init__(orientation="vertical", **kwargs)

        self.info = Label(
            size_hint=(1, 0.12),
            text=(
                "Left click: add seed | Right click: clear | Backspace: undo | Enter: run\n"
                "The propagated mask is saved in green until the object is considered lost."
            )
        )

        self.viewer = VideoSegmentationWidget(video_path, output_path)
        self.add_widget(self.info)
        self.add_widget(self.viewer)


class VideoPropagationApp(App):
    def __init__(self, video_path: str, output_path: str = "propagated_output.mp4", **kwargs):
        super().__init__(**kwargs)
        self.video_path = video_path
        self.output_path = output_path

    def build(self):
        root = RootUI(self.video_path, self.output_path)
        Window.bind(on_key_down=root.viewer.on_key_down)
        return root


if __name__ == "__main__":
    VIDEO_PATH = "input.mp4"
    OUTPUT_PATH = "propagated_output.mp4"

    VideoPropagationApp(VIDEO_PATH, OUTPUT_PATH).run()