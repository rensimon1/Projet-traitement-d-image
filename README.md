
# Suivi d'Objet avec Optical Flow et Filtre de Kalman

Ce projet implémente un système de suivi d'objet dans une vidéo en utilisant **Harris Corner Detection**, l'**Optical Flow de Lucas-Kanade**, et un **filtre de Kalman** pour prédire et suivre la trajectoire de l'objet sélectionné. 

---

## 🚀 Fonctionnalités
- **Détection des coins (Harris Corner Detection)** pour identifier les points d'intérêt dans la région sélectionnée.
- **Suivi par Optical Flow** pour capturer le déplacement de l'objet dans les frames successives.
- **Prédiction avec Filtre de Kalman** pour améliorer la robustesse du suivi, même en cas de perte de points d'intérêt.

---

## 📁 Structure du Projet

- **`projet.py`** : Script principal pour charger une vidéo, sélectionner un objet et suivre sa trajectoire.
- **`video5.mp4`** : Exemple de vidéo utilisée pour les tests (à remplacer par votre propre vidéo).
- **`README.md`** : Document explicatif du projet.

---

## 🛠️ Pversions utilisées
1. **Python 3.12.4** 
2. Bibliothèques nécessaires :
   - OpenCV (4.10.0)
   - NumPy
   - Matplotlib

---

## Utilisation
1. Lancer projet.py
2. Clique droit : Clique pour dessiner un rectangle autour de l'objet à suivre. Pour cela, Clique droit pour séléctionner un premier point puis clique droit pour sélectionner le deuxième point diagonalement opposé. ( L'objet est maintenant encadré en vert)
3. Appuyer sur Echap pour valider la séléction
4. 2 fenêtres s'affichent pour visualiser le suivi de l'objet séléctionné avec OF et OF + KF

---

## Contribution
Ce projet a été réalisé par Ritchy AGNESA et Simon REN.

