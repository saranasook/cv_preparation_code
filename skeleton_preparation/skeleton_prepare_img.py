import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# --------‑‑‑‑ Configuration ‑‑‑‑-------- (example) #
MAX_POINTS                          = 8
VISIBILITY                          = 2
CONNECTECTIONS                      = [(1,2), (2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(1,8),(2,7),(3,6)]
SUPPORTED_EXT                       = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

img_dir                             = "your_image_folder_directory"
label_dir                           = "your_output_folder_directory"
recursive                           = False
class_id                            = 0
img_dir                             = Path(img_dir)
label_dir                           = Path(label_dir)


def normalize(val, max_val):
    return val/max_val

def create_yolo_pose_label(image_path:Path, kps, label_outdir:Path, class_id = 0):
    img                             = cv2.imread(str(image_path))

    if img is None:
        raise IOError(f"Cannot read {image_path}")
    
    h,w                             = img.shape[:2]

    ### bbox ###
    xs,ys                           = zip(*kps)
    xmin, xmax                      = min(xs), max(xs)
    ymin, ymax                      = min(ys), max(ys)
    xc                              = normalize((xmin + xmax) / 2, w)
    yc                              = normalize((ymin + ymax) / 2, h)
    bw                              = normalize(xmax - xmin, w)
    bh                              = normalize(ymax - ymin, h)

    #### Keypoint (x y v x 8) ### (8 keypoint as example) ####
    flat_kps                        = []
    for x,y in kps:
        flat_kps.extend([normalize(x, w), normalize(y, h), VISIBILITY])
    
    label_vals                      = [class_id,xc, yc,bw, bh] + flat_kps
    label_str                       = " ".join(map(str, label_vals))

    label_outdir.mkdir(parents = True, exist_ok = True)
    out_path                        = label_outdir / (image_path.stem + '.txt')
    out_path.write_text(label_str + "\n")
    print(f"✓ Saved {out_path.relative_to(Path.cwd())}")

def draw_skeleton(canvas, kps, highlight_last = True):

    vis                             = canvas.copy()
    ### Lines ###
    for a,b in CONNECTECTIONS:
        if a <= len(kps) and b <=len(kps):
            cv2.line(vis, kps[a-1], kps[b-1], (0,255,0),2)
    
    ### Point ##
    for i, (x,y) in enumerate(kps):
        color                       = (0,0,255) if (highlight_last and i == len(kps) - 1) else (255, 0, 0)
        cv2.circle(vis, (x,y), 4, color, -1)
    
    return vis

class Annotator:
    def __init__(self, image_path: Path):
        self.image_path             = image_path
        self.img                    = cv2.imread(str(image_path))

        if self.img is None:
            raise IOError(f"Cannot read {image_path}")
        self.kps                    = []
        self.window                 = "Annotate — " + image_path.name
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._on_click)
    
    def _on_click(self, event, x,y, flags, _params):
        ### Left click -> add
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.kps) < MAX_POINTS:
                self.kps.append((x,y))
        
        #### Right click -> undo
        if event == cv2.EVENT_RBUTTONDOWN and self.kps:
            self.kps.pop()
    
    def run(self):
        while True:
            show                    = draw_skeleton(self.img, self.kps)
            cv2.imshow(self.window, show)
            key                     = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                cv2.destroyAllWindows(self.window)
                sys.exit(0)
            if key == ord("s"):
                cv2.destroyAllWindows(self.window):
                return None
            if key == ord("r"):
                self.kps.clear()
            if key in (13,10):
                if len(self.kps) == MAX_POINTS:
                    cv2.destroyAllWindows(self.window)
                    return self.kps



####### Main code is here ############
### gather files ###
files                           = sorted(
    p for p in (img_dir.rglob("*") if recursive else img_dir.iterdir())
        if p.suffix.lower() in SUPPORTED_EXT
)

print(f"Found {len(files)} images.  Left‑click to add, right‑click to undo.")
print("Keys:  Enter=save  r=reset  s=skip  q=quit\n")

for img_path in files:
    print(f"→ {img_path.name}")
    annot                       = Annotator(img_path)
    kps                         = annot.run()
    print(kps)
    if kps:
        create_yolo_pose_label(img_path, kps, label_dir, class_id=class_id)

print("\nAll done!")
