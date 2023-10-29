import cv2
from yolov8 import YOLOv8

class Object_Detection():
    def __init__(self):
        self.model_path = "models/day09/best.onnx"
        self.yolov8_detector = YOLOv8(self.model_path, conf_thres=0.2, iou_thres=0.3)
        
    def image_object_detection(self, img):

        # Detect Objects
        boxes, scores, class_ids = self.yolov8_detector(img)

        # Draw detections
        combined_img = self.yolov8_detector.draw_detections(img)
        return combined_img
    
    # def video_object_detection(self):
    #     start_time = 5 # skip first {start_time} seconds
    #     cap = cv2.VideoCapture(self.object_url)
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))
        