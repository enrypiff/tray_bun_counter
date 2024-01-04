import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import torch


# define the video to read, because I don't have a webcam on the machine
cap = cv2.VideoCapture("PATH VIDEO")

# define the model to use. It was trained on a custom dataset annotated by myself
model = YOLO("PATH MODEL")

class_names = ["bun", "tray"]

mask = cv2.imread("mask.png")

# tracking
tracker_bun = Sort(max_age=1, min_hits=1, iou_threshold=0.4)
tracker_tray = Sort(max_age=1, min_hits=1, iou_threshold=0.4)

# coordinates of the counting line
limits = [0, 590, 380, 450]
total_count_bun = []
total_count_tray = []

# Get the Default resolutions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# define the codec and filename
out = cv2.VideoWriter('output_buns_tray15.mp4',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 15.0, (frame_width,frame_height))

prev_frame_time = 0
new_frame_time = 0

tray_captured = []

while (cap.isOpened()):
    new_frame_time = time.time()
    success, img_origin = cap.read()

    if success:

        img = img_origin.copy()
        img_height, img_width, img_color = img.shape
        img_region = cv2.bitwise_and(img, mask)

        img_graphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
        # use cvzone because it's easier to overlap two images
        img = cvzone.overlayPNG(img, img_graphics, (0, 0))
        results = model.predict(img_region.copy(), stream=True)

        # remove the logo from the video
        roi = img[0:70, 620:720]
        # applying a gaussian blur over this new rectangle area
        roi = cv2.blur(roi, (23, 23), 30)
        # impose this blurred image on original image to get final image
        img[0:0 + roi.shape[0], 620:620 + roi.shape[1]] = roi

        roi = img[340:430, 430:560]
        # applying a blur over this new rectangle area
        roi = cv2.blur(roi, (23, 23), 30)
        # impose this blurred image on original image to get final image
        img[340:340 + roi.shape[0], 430:430 + roi.shape[1]] = roi

        detections_bun = np.empty((0, 5))
        detections_tray = np.empty((0, 5))

        for r in results:
            boxes = r.boxes

            # make list
            tray_masks = []

            for i, box in enumerate(boxes):
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                current_class = class_names[cls]

                if current_class == "bun" and conf > 0.4:
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections_bun = np.vstack((detections_bun, current_array))

                if current_class == "tray" and conf > 0.8:
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections_tray = np.vstack((detections_tray, current_array))
                    print("class id n:"+str(i))

                    # Convert mask to single channel image
                    mask_raw = r.masks[i].cpu().data.numpy().transpose(1, 2, 0)
                    # Resize the mask to the same size as the image
                    mask_seg = cv2.resize(mask_raw, (img_width, img_height))

                    mask_seg = mask_seg * 255
                    tray_masks.append(mask_seg.astype(np.uint8))


        results_tracker_bun = tracker_bun.update(detections_bun)
        results_tracker_tray = tracker_tray.update(detections_tray)

        # show the line on which the counting is made
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (255, 255, 255), 5)

        # the line for the counting is inclined so we need to use the formula to get the line y=mx + q
        m = (limits[3] - limits[1]) / (limits[2] - limits[0])
        q = ((limits[2] * limits[1]) - (limits[0] * limits[3])) / (limits[2] - limits[0])

        for result in results_tracker_bun:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 255, 255), cv2.FILLED)

            yr = cx * m + q

            if limits[0] < cx < limits[2] and yr - 40 < cy < yr + 5:
                if total_count_bun.count(id) == 0:
                    total_count_bun.append(id)

                    # change color of the line and the circle when the counting is made
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        for result in results_tracker_tray:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            cx, cy = x1 + w // 2, y1 + h // 2

            yr = cx * m + q

            if limits[0] < cx < limits[2] and yr - 40 < cy < yr + 150:
                if total_count_tray.count(id) == 0:
                    total_count_tray.append(id)

                    img_tray = img_origin[y1:y2, x1:x2]
                    tray_captured.append(img_tray)

                    # change color of the line and the circle when the counting is made
                    cv2.circle(img, (cx, cy), 200, (0, 255, 0), 3)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        # write the total counting on the frame
        cv2.putText(img, str(len(total_count_tray)), (90, 50), cv2.FONT_ITALIC, 2, (0, 0, 255), 2)
        cv2.putText(img, str(len(total_count_bun)), (90, 115), cv2.FONT_ITALIC, 2, (0, 0, 255), 2)

        # merge the tray masks of the frame
        tray_mask_result = np.full((img_height, img_width), 0, dtype=np.uint8)

        for tray_mask in tray_masks:
            tray_mask_result = cv2.bitwise_or(tray_mask, tray_mask_result)
        # make a 3 channel mask
        mask_3channel = cv2.merge((tray_mask_result, tray_mask_result, tray_mask_result))

        # copy where we'll assign the new values
        red_frame = np.copy(img)
        # boolean indexing and assignment based on mask
        red_frame[(mask_3channel == 255).all(-1)] = [0, 0, 255]
        alpha = 0.7
        new_img = cv2.addWeighted(red_frame, 1 - alpha, img, alpha, 0, red_frame)

        # show the image of last tray detected
        x_offset = 0
        y_offset = 150
        last_tray_index = len(tray_captured) - 1
        if last_tray_index >= 0:
            new_img[y_offset:y_offset + tray_captured[last_tray_index].shape[0],
            x_offset:x_offset + tray_captured[last_tray_index].shape[1]] = tray_captured[last_tray_index]

        # show on screen the frame
        cv2.imshow("image", new_img)

        # write the  frame in the video
        out.write(new_img)

        # calculate the frame rate of the processed video
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print(fps)

        cv2.waitKey(1)
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
