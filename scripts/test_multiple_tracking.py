import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.synthetic_detection import BeltLocalizer

def test_single_object():
    localizer = BeltLocalizer(min_area=50, min_dim=10, pad=5, dist_threshold=100)
    frame_size = 400
    
    track_counts = {}
    frames_with_detections = 0
    
    for i in range(15):
        frame = np.ones((frame_size, frame_size, 3), dtype=np.uint8) * 60
        
        x = int((i / 14.0) * 300)
        y = 200
        frame[y:y+15, x:x+15] = 255
        
        proposals, _ = localizer.update(frame)
        
        if proposals:
            frames_with_detections += 1
            for tid, box in proposals:
                track_counts[tid] = track_counts.get(tid, 0) + 1
                
    persistent_ids = {tid for tid, count in track_counts.items() if count >= 5}
    print(f"Single object test: detected in {frames_with_detections}/15 frames.")
    print(f"Persistent track IDs (>=5 frames): {persistent_ids}")
    if len(persistent_ids) == 1:
        print("Single object test passed!")
    else:
        print("Single object test failed!")


def test_multiple_objects():
    localizer = BeltLocalizer(min_area=50, min_dim=10, pad=5, dist_threshold=100)
    frame_size = 400
    
    track_counts = {}
    frames_with_detections = 0
    
    for i in range(15):
        frame = np.ones((frame_size, frame_size, 3), dtype=np.uint8) * 60
        
        x1 = int((i / 14.0) * 300)
        y1 = 100
        frame[y1:y1+15, x1:x1+15] = 255
        
        x2 = int((i / 14.0) * 300)
        y2 = 300
        frame[y2:y2+15, x2:x2+15] = 255
        
        proposals, _ = localizer.update(frame)
        
        if proposals:
            frames_with_detections += 1
            for tid, box in proposals:
                track_counts[tid] = track_counts.get(tid, 0) + 1
                
    persistent_ids = {tid for tid, count in track_counts.items() if count >= 5}
    print(f"Multiple objects test: detected in {frames_with_detections}/15 frames.")
    print(f"Persistent track IDs (>=5 frames): {persistent_ids}")
    if len(persistent_ids) == 2:
        print("Multiple objects test passed!")
    else:
        print("Multiple objects test failed!")


if __name__ == "__main__":
    print("Running tracking tests...")
    test_single_object()
    print("-" * 30)
    test_multiple_objects()
