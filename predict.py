from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="sample_images",
    save=True,
    conf=0.25
)

print("Prediction completed. Check the runs/detect folder.")