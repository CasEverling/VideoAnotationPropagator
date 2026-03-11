import cv2
import numpy as np
from collections import deque

class RegionGrowing:
    """Implements region growing segmentation with multi-click support"""
    
    DEFAULT_COLOR_THRESHOLD = 20
    DEFAULT_GRAY_THRESHOLD = 15
    DEFAULT_MORPH_KERNEL = 5
    
    def __init__(self):
        # just setting up defaults
        self.color_threshold = RegionGrowing.DEFAULT_COLOR_THRESHOLD
        self.gray_threshold = RegionGrowing.DEFAULT_GRAY_THRESHOLD
        self.morph_kernel = RegionGrowing.DEFAULT_MORPH_KERNEL
        self.seeds = []   # list of (x,y)
        self.current_mask = None
    
    def add_seed(self, x, y):
        self.seeds.append((x,y))
    
    def clear_seeds(self):
        self.seeds = []
        self.current_mask = None
    
    def undo_last_seed(self):
        if len(self.seeds) > 0:
            self.seeds.pop()
    
    def _is_similar(self, a, b, t, is_color):
        # quick check
        if is_color:
            d = np.sqrt(((a.astype(float)-b.astype(float))**2).sum())
        else:
            d = abs(float(a)-float(b))
        return d < t
    
    def region_grow_single_seed(self, img, seed, t=None):
        h,w = img.shape[:2]
        is_color = (len(img.shape)==3)
        if t is None:
            t = self.color_threshold if is_color else self.gray_threshold
        
        masky = np.zeros((h,w), np.uint8)
        visited = np.zeros((h,w), bool)
        
        xx,yy = seed
        if xx<0 or xx>=w or yy<0 or yy>=h:
            print("bad seed:", seed)
            return masky
        
        seedval = img[yy,xx]
        q = deque()
        q.append((xx,yy))
        visited[yy,xx] = True
        
        neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        maxsz = (w*h)//2
        sz = 0
        
        while q and sz<maxsz:
            cx,cy = q.popleft()
            val = img[cy,cx]
            if self._is_similar(seedval,val,t,is_color):
                masky[cy,cx] = 255
                sz += 1
                for dx,dy in neigh:
                    nx,ny = cx+dx, cy+dy
                    if 0<=nx<w and 0<=ny<h and not visited[ny,nx]:
                        visited[ny,nx] = True
                        q.append((nx,ny))
        if sz>=maxsz:
            print("region capped at", maxsz)
        return masky
    
    def region_grow_multi_seed(self, img, seeds, t=None):
        if len(seeds)==0:
            return np.zeros(img.shape[:2], np.uint8)
        out = np.zeros(img.shape[:2], np.uint8)
        for s in seeds:
            tmp = self.region_grow_single_seed(img, s, t)
            out = cv2.bitwise_or(out,tmp)
        return out
    
    def smooth_mask(self, mask, k=None):
        if k is None: k=self.morph_kernel
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,ker)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,ker)
        return mask
    
    def extract_contours(self, mask):
        c,h = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        return c,h
    
    def get_largest_contour(self, c):
        if not c: return None
        return max(c,key=cv2.contourArea)
    
    def segment(self, img, threshold=None, smooth=True):
        if len(self.seeds) == 0:
            return np.zeros(img.shape[:2], np.uint8), []

        # pass threshold directly to the region growing
        mask = self.region_grow_multi_seed(img, self.seeds, threshold)

        if smooth:
            mask = self.smooth_mask(mask)

        contours, _ = self.extract_contours(mask)
        self.current_mask = mask
        return mask, contours

    def draw_contours_on_image(self, img, c, color=(0, 255, 0), thickness=2):
        res = img.copy()
        cv2.drawContours(res, c, -1, color, thickness)
        return res

    def create_overlay(self,img,mask,alpha=0.3,color=(0,255,0)):
        over = img.copy()
        cm = np.zeros_like(img)
        cm[mask>0] = color
        res = cv2.addWeighted(over,1-alpha,cm,alpha,0)
        return res