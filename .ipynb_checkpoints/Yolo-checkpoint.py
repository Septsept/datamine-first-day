from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # initialize model
results = model("hibiscus.jpg")  # perform inference
results[0].show()  # display results for the first image