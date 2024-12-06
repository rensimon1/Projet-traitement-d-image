import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# Charger la vidéo
video = cv2.VideoCapture("video5.mp4")
ret, frame = video.read()

# Initialisation des variables pour la sélection de la ROI
x_min, y_min, x_max, y_max = 36000, 36000, 0, 0

def coordinat_chooser(event, x, y, flags, param):
    global x_min, y_min, x_max, y_max
    if event == cv2.EVENT_RBUTTONDOWN:
        x_min = min(x, x_min)
        y_min = min(y, y_min)
        x_max = max(x, x_max)
        y_max = max(y, y_max)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
    if event == cv2.EVENT_MBUTTONDOWN:
        x_min, y_min, x_max, y_max = 36000, 36000, 0, 0

cv2.namedWindow('coordinate_screen')
cv2.setMouseCallback('coordinate_screen', coordinat_chooser)

# Sélection de la ROI
while True:
    cv2.imshow("coordinate_screen", frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        break

print(f"x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max}")

# ROI et sélection du point d'intérêt
roi_image = frame[y_min:y_max, x_min:x_max]
roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

# Détection des caractéristiques dans la ROI
feature_params = dict(maxCorners=20, qualityLevel=0.2, minDistance=7, blockSize=7)
first_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(first_gray)
mask[y_min:y_max, x_min:x_max] = 255
points = cv2.goodFeaturesToTrack(first_gray, mask=mask, **feature_params)

# Sélection d'un point d'intérêt
if points is not None and len(points) > 0:
    selected_point = points[0].ravel()
    print(f"Point sélectionné : x={selected_point[0]}, y={selected_point[1]}")
    p0 = np.array([selected_point], dtype=np.float32).reshape(-1, 1, 2)
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

if selected_point is not None:
    kalman.statePost = np.array([[selected_point[0]], [selected_point[1]], [0], [0]], dtype=np.float32)
else:
    print("Impossible d'initialiser le filtre de Kalman : Aucun point sélectionné.")
    exit()

# Lucas-Kanade Optical Flow params
lk_params = dict(winSize=(7, 7), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialisation des variables pour le suivi
ret, old_frame = video.read()
old_gray = first_gray
mask = np.zeros_like(old_frame)
frame_count = 0
start_time = time.time()

# Boucle principale pour le suivi
while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if len(good_new) > 0:
            measured = np.array([[np.float32(good_new[0][0])], [np.float32(good_new[0][1])]])
            kalman.correct(measured)
            prediction = kalman.predict()

            px, py = prediction[0], prediction[1]
            a, b = good_new[0].ravel()

            mask = cv2.line(mask, (int(px), int(py)), (int(a), int(b)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(px), int(py)), 5, (255, 0, 0), -1)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

    else:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        mask = np.zeros_like(frame)

    img = cv2.add(frame, mask)
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(img, f"FPS: {fps:.2f}", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('frame', img)

    frame_count += 1
    k = cv2.waitKey(25)
    if k == 27:
        break

cv2.destroyAllWindows()
video.release()
