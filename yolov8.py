import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

from supervision.draw.color import ColorPalette
from supervision import Detections
from supervision import BoxAnnotator
import os

class ObjectDetection:

    def __init__(self, capture_index):
       
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = BoxAnnotator(color=ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)
    

    def load_model(self):
        model = YOLO("yolov8m.pt")  # load a pretrained YOLOv8n model
        model.fuse()
    
        return model


    def predict(self, frame):
        results = self.model(frame)
        return results


    def findDistance(self, boundingBox):
        width = boundingBox[0][2] - boundingBox[0][0]
        height = boundingBox[0][3] - boundingBox[0][1]
        distance = (2 * 3.14 * 180) / (width + height * 360) * 1000 + 3

        return distance * 0.0254
    

    def plot_bboxes(self, results, frame):
        
        xyxys = []
        confidences = []
        class_ids = []
        
        distance = []
        minimumDistance = float("inf")
        # Extract detections for person class
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            
            if class_id[0] == 2:
                print("class: ", self.CLASS_NAMES_DICT[class_id[0]])

                boundingBox = result.boxes.xyxy.cpu().numpy()
                xyxys.append(boundingBox)

                confidence = result.boxes.conf.cpu().numpy()
                confidences.append(confidence)
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

                print("Coordinates of bounding box: ", boundingBox)
                print("Confidence Score: ", confidence)

                currentDistance = self.findDistance(boundingBox)
                if(currentDistance < minimumDistance):
                    minimumDistance = currentDistance

                distance.append(currentDistance)

        print("Distance: ", distance)
        print("Minimum Distance: ", minimumDistance)
        
            
        
        # Setup detections for visualization
        detections = Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                    )
        
    
        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections]
        
        # Annotate and display frame
        frame = self.box_annotator.annotate(frame, detections, self.labels)
        
        return frame
    
    
    
    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      
        while True:
          
            start_time = time()
            ret, frame = cap.read()
            assert ret
            
            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
             
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            cv2.imshow('YOLOv8 Detection', frame)
 
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
os.environ["CUDA_VISIBLE_DEVICES"] = "0"           
detector = ObjectDetection(capture_index=1)
detector()