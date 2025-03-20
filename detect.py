import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

def detect_waste_dumping():
    # Initialize models
    model_pose = YOLO('yolo11l-pose.pt')
    model_object = YOLO(r"new_garbage_model.pt")
    
    # Print available classes from your model to verify
    print(f"Available detection classes: {model_object.names}")
    
    # Open webcam
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {frame_width}x{frame_height}")
    
    # Configuration - Using lower thresholds for easier detection
    HAND_OBJECT_DISTANCE = 150 # Increased max pixel distance between hand and object
    
    # Expand LITTER_CLASSES to include more variations of waste items
    LITTER_CLASSES = ['bottle', 'bag', 'plastic bag', 'cup', 'wrapper', 'cell phone', 'trash', 'waste', 'garbage', 'litter']
    
    # Much lower threshold for drop detection
    DROP_VELOCITY_THRESHOLD = 5  # Minimum downward speed to consider as dropping
    
    # Tracking history
    hand_positions = defaultdict(list)
    object_positions = defaultdict(list)
    object_in_hand = defaultdict(bool)
    object_was_in_hand = defaultdict(bool)  # Track if object was ever in hand
    history_length = 15  # Increased history length to better track movement
    
    # Drop detection parameters
    dropping_detected = False
    drop_cooldown = 0
    DROP_COOLDOWN_FRAMES = 30  # Show drop alert for this many frames
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to receive frame")
            break
            
        frame_count += 1
        
        # Decrement cooldown counter if active
        if drop_cooldown > 0:
            drop_cooldown -= 1
            if drop_cooldown == 0:
                dropping_detected = False
        
        # Run detection
        results_pose = model_pose.predict(frame)
        results_object = model_object.predict(frame, conf=0.25)  # Lower confidence threshold
        
        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        # Draw pose keypoints and get hand positions
        current_hands = []
        if results_pose[0].keypoints is not None:
            for person_id, kps in enumerate(results_pose[0].keypoints.xy.cpu().numpy()):
                if len(kps) >= 17:  # Ensure all keypoints exist
                    # Get and draw all keypoints for visualization
                    for i, kp in enumerate(kps):
                        cv2.circle(annotated_frame, tuple(kp.astype(int)), 3, (0, 255, 0), -1)
                    
                    # Get hand positions (wrists)
                    left_wrist = kps[9].astype(float)
                    right_wrist = kps[10].astype(float)
                    
                    # Add hands to current frame's hands
                    current_hands.append(("left_" + str(person_id), left_wrist))
                    current_hands.append(("right_" + str(person_id), right_wrist))
                    
                    # Draw larger circles for hands
                    cv2.circle(annotated_frame, tuple(left_wrist.astype(int)), 8, (255, 0, 0), -1)
                    cv2.circle(annotated_frame, tuple(right_wrist.astype(int)), 8, (255, 0, 0), -1)
                    
                    # Update hand position history
                    for hand_id, hand_pos in [("left_" + str(person_id), left_wrist), 
                                             ("right_" + str(person_id), right_wrist)]:
                        hand_positions[hand_id].append(hand_pos)
                        # Keep only recent positions
                        if len(hand_positions[hand_id]) > history_length:
                            hand_positions[hand_id].pop(0)
        
        # Process object detections
        waste_detected = False
        current_dropping_action = False
        objects_in_frame = []
        
        if results_object[0].boxes is not None:
            boxes = results_object[0].boxes.xyxy.cpu().numpy()
            classes = results_object[0].boxes.cls.cpu().numpy()
            confidences = results_object[0].boxes.conf.cpu().numpy()
            
            for idx, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                if conf < 0.25:  # Lower threshold for detection
                    continue
                    
                class_idx = int(cls)
                class_name = model_object.names[class_idx]
                
                # For debugging
                if frame_count % 60 == 0:
                    print(f"Processing {class_name} with conf {conf:.2f}")
                
                # Check if this is a waste item
                is_waste = any(waste_term in class_name.lower() for waste_term in LITTER_CLASSES)
                
                x1, y1, x2, y2 = box.astype(int)
                obj_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                obj_id = f"{class_name}_{idx}"
                
                # Focus specifically on waste items
                if is_waste:
                    waste_detected = True
                    objects_in_frame.append(obj_id)
                    
                    # Update object position history
                    object_positions[obj_id].append(obj_center)
                    if len(object_positions[obj_id]) > history_length:
                        object_positions[obj_id].pop(0)
                    
                    # Check if object is close to any hand
                    near_hand = False
                    for hand_id, hand_pos in current_hands:
                        distance = np.linalg.norm(hand_pos - obj_center)
                        if distance < HAND_OBJECT_DISTANCE:
                            near_hand = True
                            
                            # Mark object as being in hand (now and historically)
                            object_in_hand[obj_id] = True
                            object_was_in_hand[obj_id] = True
                            
                            # Draw line connecting hand and object
                            cv2.line(annotated_frame, tuple(hand_pos.astype(int)), 
                                   tuple(obj_center.astype(int)), (0, 255, 255), 2)
                            break
                    
                    # Check for dropping action - improved logic
                    # Only check objects that were previously in hand but now aren't
                    if object_was_in_hand[obj_id] and not near_hand:
                        # Calculate vertical movement from position history
                        if len(object_positions[obj_id]) >= 3:
                            # Get last few positions
                            recent_positions = object_positions[obj_id][-5:]
                            if len(recent_positions) >= 2:
                                # Calculate vertical velocities (y increases downward)
                                y_velocities = []
                                for i in range(1, len(recent_positions)):
                                    y_velocity = recent_positions[i][1] - recent_positions[i-1][1]
                                    y_velocities.append(y_velocity)
                                
                                # If mostly moving downward
                                if sum(v > 0 for v in y_velocities) > len(y_velocities) / 2:
                                    # And average downward velocity exceeds threshold
                                    avg_y_velocity = sum(y_velocities) / len(y_velocities)
                                    if avg_y_velocity >= DROP_VELOCITY_THRESHOLD:
                                        current_dropping_action = True
                                        
                                        # Debug info when dropping is detected
                                        print(f"Drop detected! Velocities: {y_velocities}, Avg: {avg_y_velocity}")
                                        
                                        # Set cooldown timer
                                        dropping_detected = True
                                        drop_cooldown = DROP_COOLDOWN_FRAMES
                    
                    # Update in-hand status (but maintain object_was_in_hand)
                    object_in_hand[obj_id] = near_hand
                    
                    # Set box color based on status
                    if dropping_detected or current_dropping_action:
                        color = (0, 0, 255)  # Red for dropping
                        cv2.putText(annotated_frame, "DROPPING!", (x1, y1 - 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    elif near_hand:
                        color = (0, 255, 0)  # Green for in hand
                    else:
                        color = (255, 0, 0)  # Blue for detected waste
                        
                    # Draw bounding box and label for waste items
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{class_name}", (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw object trajectory if tracked
                    if len(object_positions[obj_id]) > 1:
                        for i in range(1, len(object_positions[obj_id])):
                            start = object_positions[obj_id][i-1].astype(int)
                            end = object_positions[obj_id][i].astype(int)
                            cv2.line(annotated_frame, tuple(start), tuple(end), (255, 0, 255), 1)
                else:
                    # For non-waste items, just draw a simple box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                    cv2.putText(annotated_frame, f"{class_name}", (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Clean up objects not seen in this frame
        for obj_id in list(object_in_hand.keys()):
            if obj_id not in objects_in_frame:
                object_in_hand[obj_id] = False
        
        # Display status with large red alert for dropping
        if dropping_detected:
            status = "DROPPING WASTE DETECTED!"
            status_color = (0, 0, 255)  # Red
            
            # Draw attention-grabbing alert
            cv2.putText(annotated_frame, status, (20, 70), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
            
            # Draw flashing border
            if frame_count % 6 < 3:  # Flash effect
                cv2.rectangle(annotated_frame, (5, 5), (frame_width-5, frame_height-5), 
                             (0, 0, 255), 8)
        elif waste_detected:
            status = "Waste Objects Detected"
            status_color = (255, 165, 0)  # Orange
            cv2.putText(annotated_frame, status, (20, 40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        else:
            status = "Monitoring..."
            status_color = (0, 255, 0)  # Green
            cv2.putText(annotated_frame, status, (20, 40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Show frame count and detection stats
        info_text = f"Frame: {frame_count} | Waste Objects: {len(objects_in_frame)}"
        cv2.putText(annotated_frame, info_text, (20, frame_height - 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Smart Waste Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

detect_waste_dumping()