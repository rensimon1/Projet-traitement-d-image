# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 18:06:20 2024

@author: CYTech Student
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

# Charger la vidéo
video = cv2.VideoCapture("video6.mp4")
ret, first_frame = video.read()

x_min, y_min, x_max, y_max = 36000, 36000, 0, 0

# Fonction pour choisir la région d'intérêt (ROI)
def coordinat_chooser(event, x, y, flags, param):
    global x_min, y_min, x_max, y_max

    # Clic droit : sélectionner les coordonnées
    if event == cv2.EVENT_RBUTTONDOWN:
        x_min = min(x, x_min)
        y_min = min(y, y_min)
        x_max = max(x, x_max)
        y_max = max(y, y_max)
        # Dessiner un rectangle autour de la ROI
        cv2.rectangle(first_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    # Clic du milieu : réinitialiser les coordonnées
    if event == cv2.EVENT_MBUTTONDOWN:
        print("Reset coordinate data")
        x_min, y_min, x_max, y_max = 36000, 36000, 0, 0

cv2.namedWindow('coordinate_screen')
cv2.setMouseCallback('coordinate_screen', coordinat_chooser)

# Boucle pour afficher la sélection ROI
while True:
    cv2.imshow("coordinate_screen", first_frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:  # Touche ESC pour quitter
        cv2.destroyAllWindows()
        break

print(f"x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max}")

# Prendre la région d'intérêt (ROI)
roi_image = first_frame[y_min:y_max, x_min:x_max]

# Convertir la ROI en niveaux de gris
roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

# Paramètres pour la détection de coin (Harris Corner Detection)
feature_params = dict(maxCorners=20,
                      qualityLevel=0.2,
                      minDistance=7,
                      blockSize=7)

first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

mask = np.zeros_like(first_gray)
mask[y_min:y_max, x_min:x_max] = 255

# Initialisation du filtre de Kalman
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32)*1
# kalman.errorCovPost = np.eye(4, dtype=np.float32)
kalman.errorCovPost = np.eye(4, dtype=np.float32) * 0.1  # Faible incertitude initiale
kalman.errorCovPost[2,2] = kalman.errorCovPost[3,3] = 1  # Incertitude sur les vitesses

# Détection des points d'intérêt
points = cv2.goodFeaturesToTrack(first_gray, mask=mask, **feature_params)

# Sélectionner un point d'intérêt dans la ROI
if points is not None and len(points) > 0:
    for point in points:
        x, y = point.ravel()
        if y_min <= y <= y_max and x_min <= x <= x_max:
            selected_point = point
            print("Point trouvé :", selected_point)
            break
else:
    print("Aucun point trouvé dans la ROI.")
    selected_point = None
    p0 = None


if selected_point is not None:
    p0 = np.array([selected_point], dtype=np.float32)
    kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
else:
    p0 = None
    print("Aucun point trouvé dans la bounding box.")

plt.imshow(roi_gray, cmap="gray")

# Paramètres pour Lucas-Kanade Optical Flow
lk_params = dict(winSize=(5, 5),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

width = first_frame.shape[1]
height = first_frame.shape[0]

# Masques pour OF seul et OF + KF
of_mask = np.zeros_like(first_frame)
kf_mask = np.zeros_like(first_frame)

frame_count = 0
start_time = time.time()
old_gray = first_gray

# Boucle principale 
while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if p0 is not None:
        # Calcul de l'Optical Flow
        p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params) 
        
        if p1 is not None and status is not None:
            good_new = p1[status == 1]  # p1 new position, status == 1 if p1 found
            good_old = p0[status == 1]
    
            if len(good_new) > 0:
                a, b = good_new[0].ravel()
                c, d = good_old[0].ravel()
    
                # OF seul : tracé des lignes
                of_mask = cv2.line(of_mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                frame_of = cv2.circle(frame.copy(), (int(a), int(b)), 5, (0, 255, 0), -1)
    
                # Kalman Filter
                measured = np.array([[np.float32(a)], [np.float32(b)]])
                kalman.correct(measured)
                prediction = kalman.predict()
                px, py = prediction[0][0], prediction[1][0]
    
                # OF + KF : tracé des prédictions
                kf_mask = cv2.line(kf_mask, (int(px), int(py)), (int(a), int(b)), (255, 0, 0), 2)
                frame_kf = cv2.circle(frame.copy(), (int(px), int(py)), 5, (255, 0, 0), -1)
    
                # Ajouter les masques respectifs
                img_of = cv2.add(frame_of, of_mask)
                img_kf = cv2.add(frame_kf, kf_mask)
    
                # Affichage des deux fenêtres
                cv2.imshow("OF Seul", img_of)
                cv2.imshow("OF + KF", img_kf)
    
            else:
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
                of_mask = np.zeros_like(frame)
                kf_mask = np.zeros_like(frame)
    
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
            
        else:
            print("Recalcul des points d'intérêt...")
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            mask = np.zeros_like(frame)
            
            
            
            
    else:
        print("Aucun point initial disponible pour le calcul.")
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
    frame_count += 1
    k = cv2.waitKey(25)
    if k == 27:  # Touche ESC pour quitter
        break
    


video.release()
cv2.waitKey(0)
