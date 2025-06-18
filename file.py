import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# === 1. Charger le modèle de détection de masque ===
mask_model = load_model("mask_detector.model")  

# === 2. Charger l'image ===
#image_path = "messi.jpg"  #1 mets le chemin vers ton image1
image_path = "imagemasque.jpg"  #2     
#image_path = "imagemasque.jpg"  #3
frame = cv2.imread(image_path)
(h, w) = frame.shape[:2]

# === 3. Charger le modèle MobileNet SSD pour la détection de visages ===
net = cv2.dnn.readNetFromCaffe("deploy.proto.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# === 4. Préparer l'image pour la détection ===
blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

# === 5. Parcourir les visages détectés ===
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        # S'assurer que les coordonnées sont valides
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        face_img = frame[y1:y2, x1:x2]

        if face_img.size == 0:
            continue

        # Prétraitement pour la prédiction
        face_resized = cv2.resize(face_img, (224, 224))
        face_array = img_to_array(face_resized)
        face_array = preprocess_input(face_array)
        face_array = np.expand_dims(face_array, axis=0)

        # Prédiction masque ou non
        (mask, withoutMask) = mask_model.predict(face_array)[0]
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Afficher résultat
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# === 6. Afficher le résultat (taille de fenêtre réduite) ===
frame_resized = cv2.resize(frame, (800, 600))  # Ajuster la taille de la fenêtre
cv2.imshow("Résultat", frame_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
