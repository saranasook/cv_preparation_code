
##### 

import cv2
import numpy as np 
import os 

video_path                  = "your_video_directory"
img_path                    = "your_image_directory"
output_mask_path            = "your_output_mask_image_directory"


drawing                     = False
current_points              = []
lane_index                  = 1

def draw_polygon(event, x, y , flags, param):
    global drawing, current_point, img_copy, lane_index, final_mask 

    if event == cv2.EVENT_LBUTTONDOWN:
        current_point.append((x,y))
        cv2.circle(img_copy, [np.array(current_points)], isClosed=True, color(255,0,0), thickness =2)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_point) >=3:
            cv2.polylines(img_copy, [np.array(current_points)], isClosed=True, color=(255, 0, 0), thickness=2)

            ### Fill the lane in the final mask using  lane_index ###
            cv2.fillPoly(final_mask, [np.array(current_points)], lane_index)
            print(f"Added lane {lane_index}")
            lane_index      +=1

def main_image(image_path):
    global img_copy, final_mask

    ### Read first frame ###
    cap                     = cv2.VideoCapture(video_path)
    ret, frame              = cap.read()
    cap.release()

    if not ret:
        print(f"Added lane {lane_index}")
        return

    height, width           = frame.shape[:2]
    final_mask              = np.zeros((height, width), dtype=np.uint8)
    img_copy                = frame.copy()

    cv2.namedWindow("Draw Lanes")
    cv2.setMouseCallback("Draw Lanes", draw_polygon)

    print("Left click to draw points, right click to close polygon")
    print("Press 's' to save and exit")

    while True:
        display             = img_copy.copy()
        if len(current_points) > 1:
            for i in range(len(current_points) - 1):
                cv2.line(display, current_points[i], current_points[i + 1], (0, 255, 255), 2)
        
        cv2.imshow("Draw Lanes", display)
        key                 = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            break
    
    cv2.destroyAllWindows()
    cv2.imwrite(output_mask_path, final_mask)
    print(f"Saved combined mask to {output_mask_path}")

def main_video(img_path):
    global img_copy, final_mask

    ### Read image data ###
    frame                   = cv2.imread(img_path)

    if not ret:
        print(f"Added lane {lane_index}")
        return

    height, width           = frame.shape[:2]
    final_mask              = np.zeros((height, width), dtype=np.uint8)
    img_copy                = frame.copy()

    cv2.namedWindow("Draw Lanes")
    cv2.setMouseCallback("Draw Lanes", draw_polygon)

    print("Left click to draw points, right click to close polygon")
    print("Press 's' to save and exit")

    while True:
        display             = img_copy.copy()
        if len(current_points) > 1:
            for i in range(len(current_points) - 1):
                cv2.line(display, current_points[i], current_points[i + 1], (0, 255, 255), 2)
        
        cv2.imshow("Draw Lanes", display)
        key                 = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            break
    
    cv2.destroyAllWindows()
    cv2.imwrite(output_mask_path, final_mask)
    print(f"Saved combined mask to {output_mask_path}")