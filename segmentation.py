import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import cv2
import numpy as np
from PIL import Image

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Charger le modèle et le processeur d'images
image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
model.to(device)

# Initialiser la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

# Liste des classes et des couleurs associées
class_labels = [
    "Fond", "Visage", "Sourcil gauche", "Sourcil droit", "Oeil gauche",
    "Oeil droit", "Nez", "Intérieur nez", "Lèvre supérieure", "Lèvre inférieure",
    "Dent", "Oreille gauche", "Oreille droite", "Cheveux", "Col",
    "Chapeau", "Oreille gauche (bijou)", "Oreille droite (bijou)", "Lunettes"
]

# Couleurs fixes associées à chaque classe (BGR format pour OpenCV)
class_colors = [
    (0, 0, 0),          # Fond - noir
    (255, 220, 177),    # Visage - beige
    (128, 0, 128),      # Sourcil gauche - violet
    (128, 0, 128),      # Sourcil droit - violet
    (255, 0, 0),        # Oeil gauche - rouge
    (255, 0, 0),        # Oeil droit - rouge
    (0, 255, 255),      # Nez - jaune
    (0, 128, 255),      # Intérieur nez - orange
    (0, 0, 255),        # Lèvre supérieure - bleu
    (0, 0, 255),        # Lèvre inférieure - bleu
    (255, 255, 255),    # Dent - blanc
    (0, 255, 0),        # Oreille gauche - vert clair
    (0, 255, 0),        # Oreille droite - vert clair
    (139, 69, 19),      # Cheveux - marron
    (128, 128, 128),    # Col - gris
    (255, 0, 255),      # Chapeau - rose
    (192, 192, 192),    # Oreille gauche (bijou) - gris clair
    (192, 192, 192),    # Oreille droite (bijou) - gris clair
    (0, 255, 255),      # Lunettes - cyan
]

# Conversion en tableau numpy pour correspondre aux indices
class_colors = np.array(class_colors, dtype=np.uint8)

while True:
    # Lire une image de la webcam
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire l'image de la webcam.")
        break

    # Convertir l'image de BGR (OpenCV) en RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convertir en PIL.Image
    pil_image = Image.fromarray(rgb_frame)

    # Préparer les entrées pour le modèle
    inputs = image_processor(images=pil_image, return_tensors="pt").to(device)

    # Effectuer la segmentation
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Ajuster la taille des résultats au format d'entrée (dimension H x W)
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=pil_image.size[::-1],  # Hauteur, Largeur
        mode="bilinear",
        align_corners=False
    )

    # Obtenir les étiquettes prédominantes (classes)
    labels = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    # Créer une image colorée basée sur les classes avec les couleurs fixes
    segmented_image = class_colors[labels]

    # Superposer le masque segmenté sur l'image originale
    overlay = cv2.addWeighted(frame, 0.5, segmented_image, 0.5, 0)

    # Ajouter une légende dans le coin supérieur gauche
    legend_x, legend_y = 10, 10
    for idx, label in enumerate(class_labels):
        color = [int(c) for c in class_colors[idx]]
        cv2.rectangle(overlay, (legend_x, legend_y + idx * 25), (legend_x + 20, legend_y + idx * 25 + 20), color, -1)
        cv2.putText(overlay, label, (legend_x + 30, legend_y + idx * 25 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Afficher l'image avec la segmentation et la légende
    cv2.imshow("Segmented Frame with Legend", overlay)

    # Quitter si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
