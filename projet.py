# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 18:06:20 2024

@author: CYTech Student
"""


import cv2
import matplotlib.pyplot as plt 
import numpy as np
import time

video= cv2.VideoCapture("video.mp4")
ret,first_frame = video.read()

x_min,y_min,x_max,y_max=36000,36000,0,0

def coordinat_chooser(event,x,y,flags,param):
    global go , x_min , y_min, x_max , y_max

    # when you click the right button, it will provide coordinates for variables
    if event==cv2.EVENT_RBUTTONDOWN:
        
        x_min=min(x,x_min) 
        y_min=min(y,y_min)
        x_max=max(x,x_max)
        y_max=max(y,y_max)
        
        

        # draw rectangle
        cv2.rectangle(first_frame,(x_min,y_min),(x_max,y_max),(0,255,0),1)

    if event==cv2.EVENT_MBUTTONDOWN:
        print("reset coordinate  data")
        x_min,y_min,x_max,y_max=36000,36000,0,0

cv2.namedWindow('coordinate_screen')
# Set mouse handler for the specified window, in this case, "coordinate_screen" window
cv2.setMouseCallback('coordinate_screen',coordinat_chooser)
    


while True:
    cv2.imshow("coordinate_screen",first_frame) # show only first frame 
    
    k = cv2.waitKey(5) & 0xFF # after drawing rectangle press ESC   
    if k == 27:
        cv2.destroyAllWindows()
        break



print(f"x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max}")


# take region of interest ( take inside of rectangle )
roi_image=first_frame[y_min:y_max,x_min:x_max]

# convert roi to grayscale
roi_gray=cv2.cvtColor(roi_image,cv2.COLOR_BGR2GRAY) 

# Params for corner detection
feature_params = dict(maxCorners=20,  # We want only one feature
                      qualityLevel=0.2,  # Quality threshold 
                      minDistance=7,  # Max distance between corners, not important in this case because we only use 1 corner
                      blockSize=7)

first_gray = cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(first_gray)
mask[y_min:y_max, x_min:x_max] = 255

# Harris Corner detection
points = cv2.goodFeaturesToTrack(first_gray, mask=mask, **feature_params)



        
# Sélection d'un point d'intérêt
if points is not None and len(points) > 0:
    
    for point in points:
        x, y = point.ravel()
        print(f"x: {x}, y: {y}")
        if y_min <= y <= y_max and x_min <= x <= x_max: #verify if point is in ROI
            selected_point = point
            print("Point trouvé :", selected_point)
            break

else:
    print("Aucun point trouvé dans la ROI.")
    selected_point = None
    p0 = None
    
# Initialisation du filtre de Kalman
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
kalman.errorCovPost = np.eye(4, dtype=np.float32)

# If a point is found, convert it to the correct shape
if selected_point is not None:
   p0 = np.array([selected_point], dtype=np.float32)
   kalman.statePost= np.array([[x], [y], [0], [0]], dtype=np.float32)
   
else:
   p0=None
   print("Aucun point trouvé dans la bounding box.")

    
plt.imshow(roi_gray,cmap="gray")







# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(7, 7),  # Window size
                 maxLevel=2,  # Number of pyramid levels
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))




width = first_frame.shape[1]
height = first_frame.shape[0]

mask = np.zeros_like(first_frame)



frame_count = 0
start_time = time.time()

old_gray = first_gray

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if p0 is not None:
        # Calculate optical flow
        p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)  # p1 new position, status == 1 if p1 found
        good_new = p1[status == 1]  
        good_old = p0[status == 1]


        if len(good_new) > 0:
            
            measured = np.array([[np.float32(good_new[0][0])], [np.float32(good_new[0][1])]])
            kalman.correct(measured)
            prediction = kalman.predict()
            px = prediction[0][0]
            py = prediction[1][0]
            
            
            
            # Calculate movement
            a, b = good_new[0].ravel()
            c, d = good_old[0].ravel()
 
            # Draw the tracks
            # mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            mask = cv2.line(mask, (int(px), int(py)),(int(a), int(b)), (0, 255, 0), 2)
            # frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)
            frame = cv2.circle(frame, (int(px), int(py)), 5, (255, 0, 0), -1)

            

        else:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            mask = np.zeros_like(frame)

        # Check if the tracked point is out of frame
        if not (25 <= a < width and 25 <= b < height):
            p0 = None  # Reset p0 to None to detect new feature in the next iteration
            selected_point_distance = 0  # Reset selected point distance when new point is detected

        img = cv2.add(frame, mask)

        # Calculate and display FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(img, f"FPS: {fps:.2f}", (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('frame', img)

        # Update previous frame and points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

   
 
    frame_count += 1

    k = cv2.waitKey(25)
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()

