from ultralytics import YOLO

# 1. Load a pretrained YOLOv11 model (n = nano, s = small, m = medium)
# We use v11n for speed on the Raspberry Pi
model = YOLO("yolo11n.pt") 

# 2. Train the model
# Replace 'data.yaml' with the path to your config file
results = model.train(
    data="data.yaml", 
    epochs=100, 
    imgsz=640, 
    batch=16, 
    device=0, # Use device=0 for GPU, device='cpu' for CPU
    project="printer_defect_detection",
    name="yolo11_baseline"
)

# 3. Validate the model
metrics = model.val()

# 4. Export the model to ONNX (optional, for better performance on Pi)
# path = model.export(format="onnx")
