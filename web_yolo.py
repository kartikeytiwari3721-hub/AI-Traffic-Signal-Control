import cv2 as cv
import numpy as np
import os
from ultralytics import YOLO

def yield_frames_from_video():
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'videos/road.mp4')
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        return

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov8n.pt')
    model = YOLO(model_path) 
    
    fps = cap.get(cv.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    frame_interval = int(fps * 0.1)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue
            
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)  

        if frame_count % frame_interval == 0:
            results = model(frame)
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    label = result.names[cls_id]
                    if label in ['car', 'truck', 'bus', 'motorbike']:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            frame = cv.resize(frame, (300, 420))
            ret, buffer = cv.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        frame_count += 1

def generate_frames():
    return yield_frames_from_video()
