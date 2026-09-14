from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Run object detection
results = model("test.jpg")

# Print detected objects
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = result.names[class_id]

        print(
            f"{class_name}: {confidence:.2%}"
        )

    # Save image with bounding boxes
    result.save(filename="result.jpg")

print("\nDetection complete!")
print("Result saved as result.jpg")