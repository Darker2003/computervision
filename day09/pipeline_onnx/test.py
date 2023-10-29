import cv2
import time
import pandas as pd
import onnxruntime
import numpy as np

# Define the path to your ONNX model
onnx_model_path = "models/day09/best.onnx"

# Load the ONNX model
ort_session = onnxruntime.InferenceSession(onnx_model_path)

cap = cv2.VideoCapture("road.mp4")
pTime = 0
fps_list = []
threshold = 0.4
i = 0

while True:
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    success, img = cap.read()
    
    if not success:
        break

    # Preprocess the image (resize, normalize, and convert to NumPy array)
    img = cv2.resize(img, (640, 640))  # Adjust the dimensions as needed
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))  # Change the order to (C, H, W)
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Perform inference with the ONNX model
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outputs = ort_session.run(None, ort_inputs)

    # Process the model outputs (assuming ort_outputs is a list of detected objects)
    for box in ort_outputs:
        print(len(box[0][0]))
        x1, y1, x2, y2, score, class_id = box[0][0]
        if score > threshold:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2), (0, 255, 0), 4))
            cv2.putText(img, f"Class {int(class_id)}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cTime = time.time()
    fps1 = 1 / (cTime - pTime)
    pTime = cTime
    fps_list.append(fps1)
    cv2.putText(img, f"FPS: {int(fps1)}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    i += 1
    if i > 1000:
        break

    # Show the image and update the FPS counter
    cv2.imshow("img", img)
    cv2.waitKey(1)

data = pd.DataFrame(fps_list)
data.to_csv("fps.csv")
