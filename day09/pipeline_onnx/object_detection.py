import cv2
import onnxruntime
import numpy as np
import torch
import torchvision.transforms as transforms

def load_onnx_model(path_onnx, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
    ort_session = onnxruntime.InferenceSession(path_onnx, providers=providers)
    return ort_session

def onnx_infer(ort_session, input_data):
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_output = ort_session.run(None, ort_inputs)
    return ort_output

# Load the YOLOv8 ONNX model
ort_session = load_onnx_model("models/day09/best.onnx")

expected_height = 640
expected_width = 640
transform = transforms.ToTensor()
# Load the video
cap = cv2.VideoCapture("road.mp4")

# Loop over the video frames
while True:

    # Capture the next frame
    ret, frame = cap.read()

    # If the frame is not empty, perform object detection
    if ret:

        # Preprocess the frame
        img = cv2.resize(frame, (expected_width, expected_height))

        frame_tensor = transform(img)

        # Convert the PyTorch tensor to a NumPy array
        frame_np = frame_tensor.numpy()

        # Make sure the image is in the expected format (batch size, channels, height, width)
        frame_np = np.expand_dims(frame_np, axis=0)

        # Perform object detection
        ort_output = onnx_infer(ort_session, frame_np)

        # Draw the bounding boxes on the frame
        for detection in ort_output[0]:

            # Get the bounding box coordinates
            x, y, w, h = detection[0:4]
            print((x,y,w,h))

            # Draw the bounding box
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Object Detection', frame)

        # If the 'q' key is pressed, quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        # If the video has ended, break the loop
        break

# Release the video capture object
cap.release()

# Close all windows
cv2.destroyAllWindows()
