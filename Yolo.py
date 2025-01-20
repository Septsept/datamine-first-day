from ultralytics import YOLO
import cv2

model = YOLO("yolov8n-oiv7.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire l'image de la webcam.")
        break

    results = model.predict(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("frame", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
