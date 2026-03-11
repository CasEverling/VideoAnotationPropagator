import cv2
import numpy as np
import os

from typing import Iterable, List, Optional

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.metrics import dp

import threading

from mobileapp.my_widgets.segmentation_picker import SegmentationPicker
from modules.cv_modules.propagation import SegmentationPropagator
from modules.cv_modules.segmented_image import SegmentedImage
from modules.cv_modules.segmentation import RegionGrowing

class MyApp(App):
    def build(self):
        self.cap = cv2.VideoCapture('input.mp4')
        self.frames: list[SegmentedImage] = []

        n = 0
        while self.cap.isOpened() and n < 100:
            n += 1
            _, frame = self.cap.read()

            if frame is None:
                break
                
            self.frames.append(
                SegmentedImage(frame)
                )
            
        self.cap.release()

        if (len(self.frames) == 0):
            raise Exception("Video not found")

        self.layout = BoxLayout(orientation = "vertical")
        self.segmentation_picker = SegmentationPicker(
            image = self.frames[0].img
        )
        self.button = Button(
            text = "Propagate to video",
            on_press = self.make_video
            )
        self.button.size_hint_y = None
        self.button.height = dp(50)

        self.layout.add_widget(self.segmentation_picker)
        self.layout.add_widget(self.button)
        
        return self.layout
    
    def make_video(self, *args, **kwgars):
        threading.Thread(
            target = self.make_video_worker, args=(self,)
        ).start()

    def make_video_worker(self, *args, **kwargs):
        mask_duration = 2
        self.button.text = "Propagating to video"
        propagator = SegmentationPropagator()

        # Create output folder if it doesn't exist
        os.makedirs("output", exist_ok=True)

        picker_mask = self.segmentation_picker.mask
        base_frame = SegmentedImage(
            self.segmentation_picker.original_image,
            picker_mask,
            picker_mask.astype(np.float32) 
        )

        marked_frames = propagator.propagate_video(
            base_frame, [frame.img for frame in self.frames[::mask_duration]]
        )

        processed_frames: list[SegmentedImage] = []
        for n, frame in enumerate(self.frames):
            chunk = marked_frames[n // mask_duration] 
            processed_frames.append(
                SegmentedImage(
                    frame.img,
                    chunk.mask,
                    chunk.mask.astype(np.float32)
                )
            )

        video = cv2.VideoWriter(
            "output.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.cap.get(cv2.CAP_PROP_FPS),
            (marked_frames[0].img.shape[1], marked_frames[0].img.shape[0])
        )

        for i, frame in enumerate(processed_frames):
            masked_image = frame.get_masked_image()
            video.write(masked_image)
            # Save each frame as a PNG to the output folder
            cv2.imwrite(f"output/frame_{i:04d}.png", masked_image)

        video.release()
        self.button.text = "Video Saved"

if __name__ == "__main__":
    MyApp().run()
