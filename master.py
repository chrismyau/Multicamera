import objdetection
import videostream
import cv2
from ultralytics import YOLO
import numpy as np
import torch
import torchvision.transforms.v2 as v2

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = YOLO('yolov8n.pt').to(device)

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

    #predefine torch functions
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    resize_transform = v2.Resize((640,640))
    

    for frame in videostream.stream_video(0):
        height, width = frame.shape[:2]
        scale_x = width / 640.0
        scale_y = height / 640.0

        np_frame = np.asarray(frame)
        
        if render % skip == 0:
            
            tensor = transform(np_frame)
            batch_transform = torch.unsqueeze(resize_transform(tensor), 0)
            detections = model(batch_transform)[0]

        # Move detections back to CPU for further processing
        boxes = detections.boxes.cpu()  # Bounding box coordinates
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
            ox1, oy1, ox2, oy2, confidence, label = box.tolist()
            if confidence > CONFIDENCE_THRESHOLD:
                global_confidences.append(confidence)
                global_detected_objects.add(label)

                x1 = int(ox1 * scale_x)
                y1 = int(oy1 * scale_y)
                x2 = int(ox2 * scale_x)
                y2 = int(oy2 * scale_y)

                if show_bounding_boxes:
                    

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, 2)
                    cv2.putText(frame, f"global {labels[label]} {round(confidence * 100, 2)}%" , 
                                (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2)

                obj_index = 0
                if focus_budget > 0 and render % skip == 0:
                    
                    # Extract the detected object as a subpatch
                    subpatch = frame[int(y1):int(y2), int(x1):int(x2)]
                    np_subpatch = np.asarray(frame)
                    tensor = transform(np_subpatch)
                    batch_transform = torch.unsqueeze(resize_transform(tensor), 0)
                    sub_detections = model(batch_transform)[obj_index]

                    for sub_box in sub_detections.boxes.data:
                        sx1, sy1, sx2, sy2, sub_confidence, sub_label = sub_box.tolist()
                        #if sub_confidence > CONFIDENCE_THRESHOLD:
                        #object_area = (sx2 - sx1) * (sy2 - sy1)  # Calculate the area of the object
                        #size_ratio = object_area / frame_area  # Calculate object size relative to frame
                        # Adjust subpatch coordinates to fit back into the original frame
                        sx1 = int(sx1 * scale_x)
                        sy1 = int(sy1 * scale_y)
                        sx2 = int(sx2 * scale_x)
                        sy2 = int(sy2 * scale_y)

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
