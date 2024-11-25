import cv2
import matplotlib.pyplot as plt 
import numpy as np

video= cv2.VideoCapture("video.mp4")
ret,frame = video.read()
# cv2.imshow("Frame", frame)
# cv2.waitKey(0)


#     _, frame= video.read()
#     cv2.imshow("Frame", frame)
#     cv2.waitKey(0)

# I am giving  big random numbers for x_min and y_min because if you initialize them as zeros whatever coordinate you go minimum will be zero 
x_min,y_min,x_max,y_max=36000,36000,0,0

def coordinat_chooser(event,x,y,flags,param):
    global go , x_min , y_min, x_max , y_max

    # when you click the right button, it will provide coordinates for variables
    if event==cv2.EVENT_RBUTTONDOWN:
        
        # if current coordinate of x lower than the x_min it will be new x_min , same rules apply for y_min 
        x_min=min(x,x_min) 
        y_min=min(y,y_min)

         # if current coordinate of x higher than the x_max it will be new x_max , same rules apply for y_max
        x_max=max(x,x_max)
        y_max=max(y,y_max)

        # draw rectangle
        cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(0,255,0),1)

    if event==cv2.EVENT_MBUTTONDOWN:
        print("reset coordinate  data")
        x_min,y_min,x_max,y_max=36000,36000,0,0

cv2.namedWindow('coordinate_screen')
# Set mouse handler for the specified window, in this case, "coordinate_screen" window
cv2.setMouseCallback('coordinate_screen',coordinat_chooser)


while True:
    cv2.imshow("coordinate_screen",frame) # show only first frame 
    
    k = cv2.waitKey(5) & 0xFF # after drawing rectangle press ESC   
    if k == 27:
        cv2.destroyAllWindows()
        break


cv2.destroyAllWindows()




# take region of interest ( take inside of rectangle )
roi_image=frame[y_min:y_max,x_min:x_max]

# convert roi to grayscale
roi_gray=cv2.cvtColor(roi_image,cv2.COLOR_BGR2GRAY) 

# Params for corner detection
feature_params = dict(maxCorners=20,  # We want only one feature
                      qualityLevel=0.2,  # Quality threshold 
                      minDistance=7,  # Max distance between corners, not important in this case because we only use 1 corner
                      blockSize=7)

first_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

# Harris Corner detection
points = cv2.goodFeaturesToTrack(first_gray, mask=None, **feature_params)


# Filter the detected points to find one within the bounding box
for point in points:
    x, y = point.ravel()
    if y_min <= y <= y_max and x_min <= x <= x_max:
        selected_point = point
        break

# If a point is found, convert it to the correct shape
if selected_point is not None:
    p0 = np.array([selected_point], dtype=np.float32)

plt.imshow(roi_gray,cmap="gray")