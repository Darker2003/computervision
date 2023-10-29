![! ONNX YOLOv8 Object Detection](https://github.com/Darker2003/computervision/blob/main/images/day9result.png)

# Important
- The input images are directly resized to match the input size of the model. I skipped adding the pad to the input image, it might affect the accuracy of the model if the input image has a different aspect ratio compared to the input size of the model. Always try to get an input size with a ratio close to the input images you will use.


### Training model
You can train and convert to onnx by using
`yolov8-training-practice.ipynb`


# Examples

 * **Image inference**:
 ```shell
 python image_object_detection.py
 ```

 * **Webcam inference**:
 ```shell
 python webcam_object_detection.py
 ```

 * **Video inference**:
 ```shell
 python video_object_detection.py
 ```

# Streamlit:
 ```shell
 streamlit run day09/yolo_api/app.py --server.fileWatcherType=none
 ```
