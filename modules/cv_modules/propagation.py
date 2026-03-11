import cv2
import numpy as np
import random
from typing import Iterable, List, Optional, Tuple

from .segmentation import RegionGrowing
from .segmented_image import SegmentedImage


class SegmentationPropagator:

    DEFAULT_MATCH_RATIO = 0.85
    DEFAULT_MIN_MATCHES = 3
    DEFAULT_NUM_CANDIDATES = 5

    def __init__(
        self,
        segmenter: Optional[RegionGrowing] = None,
        sift_params: Optional[dict] = None,
        temporal_prev_weight: float = 0.6,
        temporal_new_weight: float = 0.4,
        min_threshold: float = 127.0,
        num_candidates: int = DEFAULT_NUM_CANDIDATES,
        min_matches: int = DEFAULT_MIN_MATCHES,
        central_keep_ratio: float = 0.9,
        area_tolerance: float = 0.6,
        max_seed_points: int = 128,
    ):

        self.segmenter = segmenter or RegionGrowing()

        default_sift = dict(
            nfeatures=10000,
            contrastThreshold=0.08,
            edgeThreshold=10,
        )

        self.detector = cv2.SIFT_create(**(sift_params or default_sift))

        self.temporal_prev_weight = temporal_prev_weight
        self.temporal_new_weight = temporal_new_weight
        self.min_threshold = min_threshold

        self.num_candidates = num_candidates
        self.min_matches = min_matches

        self.central_keep_ratio = central_keep_ratio
        self.area_tolerance = area_tolerance
        self.max_seed_points = max_seed_points

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def propagate(self, prev: SegmentedImage, next_img: np.ndarray) -> SegmentedImage:

        prev_gray = self._to_gray(prev.img)
        next_gray = self._to_gray(next_img)

        prev_mask = self._ensure_binary_mask(prev.mask, prev_gray.shape)
        prev_score = self._ensure_score_mask(prev.score_mask, prev_mask)

        kp1, d1 = self._extract_object_features(prev_gray, prev_mask)
        kp2, d2 = self.detector.detectAndCompute(next_gray, None)

        if d1 is None or d2 is None:
            return self._fallback(prev_mask, prev_score, next_img)

        matches = self._match_features(d1, d2)

        if len(matches) < self.min_matches:
            return self._fallback(prev_mask, prev_score, next_img)

        candidates = self._estimate_transform_candidates(kp1, kp2, matches)

        if not candidates:
            return self._fallback(prev_mask, prev_score, next_img)

        prev_area = np.count_nonzero(prev_mask)

        best_candidate = None
        best_error = float("inf")

        for M, _, _ in candidates:

            warped_mask = self._warp_mask(prev_mask, M, next_gray.shape)
            warped_score = self._warp_score_mask(prev_score, M, next_gray.shape)

            seeds = self._spawn_region_seeds(warped_mask)

            if not seeds:
                continue

            new_mask = self.segmenter.region_grow_multi_seed(next_img, seeds)
            new_mask = self.segmenter.smooth_mask(new_mask)

            new_area = np.count_nonzero(new_mask)

            area_error = abs(new_area - prev_area) / max(prev_area, 1)

            if area_error < best_error:
                best_error = area_error
                best_candidate = (warped_mask, warped_score, new_mask)

            if area_error <= self.area_tolerance:
                break

        if best_candidate is None:
            return self._fallback(prev_mask, prev_score, next_img)

        warped_mask, warped_score, new_mask = best_candidate

        fused_score = self._temporal_fuse(warped_score, new_mask)

        final_mask = np.where(
            fused_score >= self.min_threshold,
            255,
            0,
        ).astype(np.uint8)

        return SegmentedImage(next_img, final_mask, fused_score)

    def propagate_video(
        self,
        base_frame: SegmentedImage,
        frames: Iterable[np.ndarray],
    ) -> List[SegmentedImage]:

        results = [base_frame]
        prev = base_frame

        for n, frame in enumerate(frames):
            print(f"Processed frame {n}")
            prev = self.propagate(prev, frame)
            results.append(prev)

        return results

    # ---------------------------------------------------------
    # FEATURE EXTRACTION
    # ---------------------------------------------------------

    def _extract_object_features(self, img, mask):

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(mask, kernel, iterations=1)

        return self.detector.detectAndCompute(img, dilated)

    def _match_features(self, desc1, desc2):

        matches = []

        for i, d1 in enumerate(desc1):

            distances = ((desc2 - d1) ** 2).sum(axis=1)

            best_idx = distances.argmin()
            best = distances[best_idx]

            distances[best_idx] = np.inf
            second = distances.min()

            if second > 0 and best / np.sqrt(second) < 0.85:
                matches.append((np.sqrt(best), i, best_idx))

        unique = {}

        for dist, q, t in matches:
            if t not in unique or dist < unique[t][0]:
                unique[t] = (dist, q, t)

        good = []

        for dist, q, t in unique.values():
            d = cv2.DMatch()
            d.queryIdx = q
            d.trainIdx = t
            d.distance = dist
            good.append(d)

        return sorted(good, key=lambda m: m.distance)

    # ---------------------------------------------------------
    # RANSAC TRANSFORM
    # ---------------------------------------------------------

    def _estimate_transform_candidates(self, kp1, kp2, matches):

        if len(matches) < self.min_matches:
            return []

        match_list = [(m.distance, m.queryIdx, m.trainIdx) for m in matches]

        candidates = []

        for _ in range(self.num_candidates):

            best_inliers = 0
            best_matrix = None

            for _ in range(100):

                idxs = random.sample(range(len(match_list)), 3)

                src = [kp1[match_list[i][1]].pt for i in idxs]
                dst = [kp2[match_list[i][2]].pt for i in idxs]

                M = cv2.getAffineTransform(
                    np.float32(src[:3]),
                    np.float32(dst[:3]),
                )

                inliers = 0

                for _, qs, qt in match_list:

                    p1 = np.array([*kp1[qs].pt, 1.0])
                    p2 = np.array(kp2[qt].pt)

                    pred = M @ p1

                    if np.linalg.norm(pred - p2) < 2:
                        inliers += 1

                if inliers > best_inliers:
                    best_inliers = inliers
                    best_matrix = M

            if best_matrix is not None:
                candidates.append((best_matrix, None, best_inliers))

        return sorted(candidates, key=lambda x: x[2], reverse=True)

    # ---------------------------------------------------------
    # REGION SEEDS
    # ---------------------------------------------------------

    def _spawn_region_seeds(self, warped_mask):

        ys, xs = np.where(warped_mask > 0)

        if len(xs) == 0:
            return []

        cx = xs.mean()
        cy = ys.mean()

        dx = xs - cx
        dy = ys - cy

        d2 = dx * dx + dy * dy

        order = np.argsort(d2)

        keep_n = int(len(order) * self.central_keep_ratio)

        central = order[:keep_n]

        if len(central) > self.max_seed_points:
            central = central[:: len(central) // self.max_seed_points]

        return [(int(xs[i]), int(ys[i])) for i in central]

    # ---------------------------------------------------------
    # TEMPORAL FUSION
    # ---------------------------------------------------------

    def _temporal_fuse(self, warped_prev_score, new_mask):

        new_score = (new_mask > 0).astype(np.float32) * 255.0

        fused = (
            self.temporal_prev_weight * warped_prev_score
            + self.temporal_new_weight * new_score
        )

        return np.clip(fused, 0, 255).astype(np.float32)

    # ---------------------------------------------------------
    # UTILITIES
    # ---------------------------------------------------------

    def _fallback(self, prev_mask, prev_score, next_img):

        return SegmentedImage(next_img, prev_mask.copy(), prev_score.copy())

    def _to_gray(self, img):

        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

    def _ensure_binary_mask(self, mask, shape):

        if mask is None:
            return np.zeros(shape, np.uint8)

        out = mask.copy()

        if out.shape != shape:
            out = cv2.resize(out, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

        return np.where(out > 0, 255, 0).astype(np.uint8)

    def _ensure_score_mask(self, score_mask, fallback_mask):

        if score_mask is None:
            return fallback_mask.astype(np.float32)

        out = score_mask.copy()

        if out.shape != fallback_mask.shape:
            out = cv2.resize(
                out,
                (fallback_mask.shape[1], fallback_mask.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        return out

    def _warp_mask(self, mask, M, target_shape):

        h, w = target_shape

        warped = cv2.warpAffine(mask, M, (w, h))

        return np.where(warped > 127, 255, 0).astype(np.uint8)

    def _warp_score_mask(self, score_mask, M, target_shape):

        h, w = target_shape

        return cv2.warpAffine(score_mask, M, (w, h), flags=cv2.INTER_LINEAR)