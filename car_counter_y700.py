import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

def count_vehicles_y700(video_path, output_path):
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Çizgiyi en aşağıya indir (Y = 700)
    # Not: Video yüksekliği 700'den küçükse en alt sınır olarak ayarlanır.
    line_y = min(700, height - 20) 
    line_color = (0, 0, 255) # Kırmızı
    
    vehicle_classes = [2, 3, 5, 7]
    counter = 0
    track_history = defaultdict(lambda: [])
    counted_ids = set()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
        
        # Kırmızı çizgiyi çiz
        cv2.line(frame, (0, line_y), (width, line_y), line_color, 3)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, cls in zip(boxes, ids, clss):
                if cls in vehicle_classes:
                    x, y, w, h = box
                    cx, cy = int(x), int(y)
                    
                    track = track_history[track_id]
                    track.append((cx, cy))
                    if len(track) > 30:
                        track.pop(0)
                    
                    if track_id not in counted_ids:
                        if len(track) >= 2:
                            prev_cy = track[-2][1]
                            if (prev_cy < line_y and cy >= line_y) or (prev_cy > line_y and cy <= line_y):
                                counter += 1
                                counted_ids.add(track_id)
                                cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 0), 10)
                    
                    cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (255, 255, 0), 2)
                    cv2.putText(frame, f"ID:{track_id}", (int(x-w/2), int(y-h/2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.rectangle(frame, (0, 0), (300, 80), (0, 0, 0), -1)
        cv2.putText(frame, f'SAYI: {counter}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Bitti. Y=700 çizgisini geçen: {counter}")

if __name__ == "__main__":
    count_vehicles_y700("/home/ubuntu/upload/arac.video.mp4", "/home/ubuntu/processed_y700.mp4")
