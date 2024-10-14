import objdetection
import videostream
import cv2
from ultralytics import YOLO
import numpy as np
import torch

def main():
    model = YOLO('yolov8n.pt')

    render = 0
    skip = 3  

    CONFIDENCE_THRESHOLD = 0.5
    GREEN, RED = (0, 255, 0), (0, 0, 255)
    NUM_FOCUS = 1
    SCALE = 1

    global_confidence_sum, global_confidence_count = 0, 0
    focus_confidence_sum, focus_confidence_count = 0, 0

    # Toggle flags
    show_bounding_boxes = True
    show_heatmap = True

    for frame in videostream.stream_video(0):  
        if render % skip == 0:
            frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
            detections = model(frame)[0]

        # Retrieve bounding box coordinates and class labels
        boxes = detections.boxes  # Bounding box coordinates
        labels = detections.names
        
        # Initialize an empty heatmap
        height, width, _ = frame.shape
        frame_area = height * width
        heatmap = np.zeros((height, width), dtype=np.float32)

        _, sorted_indices = torch.sort(boxes.data[:, 4], descending=False)
        sorted_boxes = boxes.data[sorted_indices]
        focus_budget = NUM_FOCUS

        global_confidences = []
        focus_confidences = []
        global_detected_objects = set()

        # Iterate over each detected object
        for box in sorted_boxes:
            x1, y1, x2, y2, confidence, label = box.tolist()
            if confidence > CONFIDENCE_THRESHOLD:
                global_confidences.append(confidence)
                global_detected_objects.add(label)
                if show_bounding_boxes:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, 2)
                    cv2.putText(frame, f"global {labels[label]} {round(confidence * 100, 2)}%" , (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2)
                
                if show_heatmap:
                    heatmap[int(y1):int(y2), int(x1):int(x2)] += confidence

                obj_index = 0
                if focus_budget > 0:
                    # Extract the detected object as a subpatch
                    subpatch = frame[int(y1):int(y2), int(x1):int(x2)]
                    sub_detections = model(subpatch)[obj_index]

                    for sub_box in sub_detections.boxes.data:
                        sx1, sy1, sx2, sy2, sub_confidence, sub_label = sub_box.tolist()
                        #if sub_confidence > CONFIDENCE_THRESHOLD:
                        #object_area = (sx2 - sx1) * (sy2 - sy1)  # Calculate the area of the object
                        #size_ratio = object_area / frame_area  # Calculate object size relative to frame

                        

                        # Adjust subpatch coordinates to fit back into the original frame
                        sx1 += x1
                        sy1 += y1
                        sx2 += x1
                        sy2 += y1
                        cv2.rectangle(frame, (int(sx1), int(sy1)), (int(sx2), int(sy2)), RED, 2)
                        cv2.putText(frame, f"focus {labels[sub_label]} {round(sub_confidence * 100, 2)}%", 
                                    (int(sx1), int(sy1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2)
                        
                        # Check if the object is detected locally but not globally
                        if sub_label not in global_detected_objects:
                            global_confidences.append(0)

                        focus_confidences.append(sub_confidence)
                            
                    focus_budget -= 1
                    obj_index += 1

        avg_global_conf = np.mean(global_confidences)
        avg_focus_conf = np.mean(focus_confidences)
        avg_conf_diff = avg_focus_conf - avg_global_conf

        # Render the average confidence difference on the top left
        cv2.putText(frame, f"Avg Conf. Diff: {round(avg_conf_diff * 100, 2)}%", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        global_confidence_sum += np.sum(global_confidences)
        global_confidence_count += len(global_confidences)

        focus_confidence_sum += np.sum(focus_confidences)
        focus_confidence_count += len(focus_confidences)

        if focus_confidence_count > 0:
            cv2.putText(frame, f"Historical Conf. Diff: {round((focus_confidence_sum/focus_confidence_count - global_confidence_sum/global_confidence_count) * 100, 2)}%", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if show_heatmap:
            heatmap = np.clip(heatmap, 0, 1)
            heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)

        # Display the result
        cv2.imshow("Image", frame)
        
        # Handle key presses
        c = cv2.waitKey(1)
        if c == 27:  # ESC key to exit
            break
        elif c == ord('b'):  # Toggle bounding box rendering
            show_bounding_boxes = not show_bounding_boxes
        elif c == ord('h'):  # Toggle heatmap rendering
            show_heatmap = not show_heatmap

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
