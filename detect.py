import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os
from datetime import datetime

def detect_waste_dumping():
    # Initialize models
    model_pose = YOLO(r'models/yolo11l-pose.pt')
    model_object = YOLO(r"models/garbage_model.pt")
    
    # Print available classes from your model to verify
    print(f"Available detection classes: {model_object.names}")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Get camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {frame_width}x{frame_height}")
    
    # Create captures directory if it doesn't exist
    captures_dir = "./captures"
    if not os.path.exists(captures_dir):
        try:
            os.makedirs(captures_dir)
            print(f"Created directory: {captures_dir}")
        except Exception as e:
            print(f"Failed to create directory {captures_dir}: {e}")
            # Fall back to current directory if we can't create /captures
            captures_dir = "./captures"
            if not os.path.exists(captures_dir):
                os.makedirs(captures_dir)
    
    # Configuration - Using lower thresholds for easier detection
    HAND_OBJECT_DISTANCE = 150 # Maximum pixel distance between hand and object
    
    # Expand LITTER_CLASSES to include more variations of waste items
    LITTER_CLASSES = ['garbage', 'paper', 'plastic']
    
    DROP_VELOCITY_THRESHOLD = 5  # Minimum downward speed to consider as dropping
    
    # Tracking history - Key change: track by person-hand-object association
    person_hand_objects = {}  # Structure: {person_id: {'left': {obj_id: history}, 'right': {obj_id: history}}}
    hand_positions = defaultdict(list)  # Structure: {hand_id: [positions]}
    object_positions = defaultdict(list)  # Structure: {obj_id: [positions]}
    
    # Track which objects were held by which hands
    object_hand_associations = {}  # Structure: {obj_id: (person_id, hand_type)}
    
    # Store person bounding boxes for capturing images
    person_bboxes = {}  # Structure: {person_id: (x1, y1, x2, y2)}
    
    history_length = 15  # Track movement over this many frames
    
    # Drop detection parameters
    dropping_detected = False
    dropping_person_id = None  # Track which person is dropping
    dropping_object_id = None  # Track which object is being dropped
    drop_cooldown = 0
    DROP_COOLDOWN_FRAMES = 30  # Show drop alert for this many frames
    
    # Image capture control
    image_captured = False  # Flag to track if we've already captured for this drop event
    
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
                dropping_person_id = None
                dropping_object_id = None
                # Reset image capture flag when drop event ends
                image_captured = False
        
        # Run detection
        results_pose = model_pose.predict(frame)
        results_object = model_object.predict(frame, conf=0.25)  # Lower confidence threshold
        
        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        # Process people and hands first
        current_hands = []  # list of (person_id, hand_type, position)
        
        # Clear person bounding boxes for this frame
        person_bboxes = {}
        
        if results_pose[0].keypoints is not None:
            # Get original prediction box coordinates for full body capture
            if hasattr(results_pose[0], 'boxes') and results_pose[0].boxes is not None:
                person_boxes = results_pose[0].boxes.xyxy.cpu().numpy()
                
                for i, box in enumerate(person_boxes):
                    if i < len(results_pose[0].keypoints.xy):
                        person_bboxes[i] = tuple(box.astype(int))
            
            for person_id, kps in enumerate(results_pose[0].keypoints.xy.cpu().numpy()):
                if len(kps) >= 17:  # Ensure all keypoints exist
                    # Initialize person data structure if needed
                    if person_id not in person_hand_objects:
                        person_hand_objects[person_id] = {'left': {}, 'right': {}}
                    
                    # Calculate person bounding box if not available from model
                    if person_id not in person_bboxes:
                        # Get min/max coordinates from keypoints to create bounding box
                        valid_kps = kps[~np.isnan(kps).any(axis=1)]
                        if len(valid_kps) > 0:
                            x_min, y_min = np.min(valid_kps, axis=0).astype(int)
                            x_max, y_max = np.max(valid_kps, axis=0).astype(int)
                            # Add some padding
                            x_min = max(0, x_min - 20)
                            y_min = max(0, y_min - 20)
                            x_max = min(frame_width, x_max + 20)
                            y_max = min(frame_height, y_max + 20)
                            person_bboxes[person_id] = (x_min, y_min, x_max, y_max)
                    
                    # Get and draw all keypoints for visualization
                    for i, kp in enumerate(kps):
                        cv2.circle(annotated_frame, tuple(kp.astype(int)), 3, (0, 255, 0), -1)
                    
                    # Get hand positions (wrists)
                    left_wrist = kps[9].astype(float)
                    right_wrist = kps[10].astype(float)
                    
                    # Add hands to current frame's hands with person association
                    current_hands.append((person_id, 'left', left_wrist))
                    current_hands.append((person_id, 'right', right_wrist))
                    
                    # Generate hand IDs
                    left_hand_id = f"left_{person_id}"
                    right_hand_id = f"right_{person_id}"
                    
                    # Draw larger circles for hands
                    cv2.circle(annotated_frame, tuple(left_wrist.astype(int)), 8, (255, 0, 0), -1)
                    cv2.putText(annotated_frame, f"P{person_id}:L", tuple(left_wrist.astype(int) + np.array([10, 0])), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                    cv2.circle(annotated_frame, tuple(right_wrist.astype(int)), 8, (255, 0, 0), -1)
                    cv2.putText(annotated_frame, f"P{person_id}:R", tuple(right_wrist.astype(int) + np.array([10, 0])), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                    # Update hand position history
                    hand_positions[left_hand_id].append(left_wrist)
                    hand_positions[right_hand_id].append(right_wrist)
                    
                    # Keep only recent positions
                    if len(hand_positions[left_hand_id]) > history_length:
                        hand_positions[left_hand_id].pop(0)
                    if len(hand_positions[right_hand_id]) > history_length:
                        hand_positions[right_hand_id].pop(0)
                    
                    # Draw person bounding box
                    if person_id in person_bboxes:
                        x1, y1, x2, y2 = person_bboxes[person_id]
                        # Draw thin rectangle around person
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                        cv2.putText(annotated_frame, f"Person {person_id}", (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
                
                # Only process waste items
                if not is_waste:
                    # For non-waste items, just draw a simple box
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                    cv2.putText(annotated_frame, f"{class_name}", (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    continue
                
                waste_detected = True
                x1, y1, x2, y2 = box.astype(int)
                obj_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                obj_id = f"{class_name}_{idx}"
                objects_in_frame.append(obj_id)
                
                # Update object position history
                object_positions[obj_id].append(obj_center)
                if len(object_positions[obj_id]) > history_length:
                    object_positions[obj_id].pop(0)
                
                # Initialize color as blue (detected waste)
                color = (255, 0, 0)
                
                # Check if object is close to any hand
                closest_hand = None
                closest_distance = float('inf')
                
                for person_id, hand_type, hand_pos in current_hands:
                    distance = np.linalg.norm(hand_pos - obj_center)
                    
                    # Find the closest hand within distance threshold
                    if distance < HAND_OBJECT_DISTANCE and distance < closest_distance:
                        closest_distance = distance
                        closest_hand = (person_id, hand_type)
                
                # Handle object-hand association
                if closest_hand:
                    person_id, hand_type = closest_hand
                    hand_pos = next(pos for pid, htype, pos in current_hands if pid == person_id and htype == hand_type)
                    
                    # Associate this object with this person's hand
                    object_hand_associations[obj_id] = (person_id, hand_type)
                    
                    # Add to person's hand-object tracking
                    person_hand_objects[person_id][hand_type][obj_id] = True
                    
                    # Draw line connecting hand and object
                    cv2.line(annotated_frame, tuple(hand_pos.astype(int)), 
                           tuple(obj_center.astype(int)), (0, 255, 255), 2)
                    
                    # Set color to green (in hand)
                    color = (0, 255, 0)
                    
                    # Add text showing which person is holding the object
                    cv2.putText(annotated_frame, f"P{person_id}:{hand_type[0].upper()}", 
                               (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Check for dropping action - only for objects that were previously associated with a hand
                elif obj_id in object_hand_associations:
                    # Get the associated person and hand
                    prev_person_id, prev_hand_type = object_hand_associations[obj_id]
                    
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
                                    # This is a drop and we know which person dropped it
                                    current_dropping_action = True
                                    dropping_detected = True
                                    dropping_person_id = prev_person_id
                                    dropping_object_id = obj_id
                                    drop_cooldown = DROP_COOLDOWN_FRAMES
                                    
                                    # Debug info when dropping is detected
                                    print(f"Drop detected by Person {prev_person_id}! Velocities: {y_velocities}, Avg: {avg_y_velocity}")
                                    
                                    # Set color to red (dropping)
                                    color = (0, 0, 255)
                                    
                                    # Mark this drop with person ID
                                    cv2.putText(annotated_frame, f"PERSON {prev_person_id} DROPPING!", 
                                               (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                    
                                    # Capture image of the person who dropped the object if:
                                    # 1. We haven't captured for this drop event yet
                                    # 2. The person is still visible in the frame
                                    if not image_captured and dropping_person_id in person_bboxes:
                                        # Get timestamp for filename
                                        timestamp = datetime.now().strftime(r"%d-%m-%y__%I.%M%p").lower()
                                        
                                        # Get person's bounding box
                                        px1, py1, px2, py2 = person_bboxes[dropping_person_id]
                                        
                                        # Ensure the bounding box is valid and within frame
                                        if px1 < px2 and py1 < py2 and px1 >= 0 and py1 >= 0 and px2 < frame_width and py2 < frame_height:
                                            # Extract person from the original frame
                                            person_img = frame[py1:py2, px1:px2].copy()
                                            
                                            # Create filename with timestamp and person ID
                                            filename = f"{captures_dir}/waste_dropper_person{dropping_person_id}_{timestamp}.jpg"
                                            
                                            # Save the image
                                            try:
                                                cv2.imwrite(filename, person_img)
                                                print(f"Saved image of person {dropping_person_id} to {filename}")
                                                # Also save full frame for context
                                                context_filename = f"{captures_dir}/waste_drop_context_{timestamp}.jpg"
                                                cv2.imwrite(context_filename, frame)
                                                image_captured = True
                                            except Exception as e:
                                                print(f"Failed to save image: {e}")
                
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
        
        # Clean up object associations for objects not in this frame
        for obj_id in list(object_hand_associations.keys()):
            if obj_id not in objects_in_frame:
                # If the object was being held and disappeared, check for potential drop
                if len(object_positions[obj_id]) > 0:
                    person_id, hand_type = object_hand_associations[obj_id]
                    # Remove from person's tracked objects
                    if person_id in person_hand_objects and hand_type in person_hand_objects[person_id]:
                        if obj_id in person_hand_objects[person_id][hand_type]:
                            del person_hand_objects[person_id][hand_type][obj_id]
                
                # After a while, remove completely from tracking
                if frame_count % 30 == 0:  # Clean up periodically
                    if obj_id in object_hand_associations:
                        del object_hand_associations[obj_id]
                    if obj_id in object_positions:
                        del object_positions[obj_id]
        
        # Display status with large red alert for dropping
        if dropping_detected and dropping_person_id is not None:
            status = f"PERSON {dropping_person_id} DROPPING WASTE DETECTED!"
            status_color = (0, 0, 255)  # Red
            
            # Draw attention-grabbing alert
            cv2.putText(annotated_frame, status, (20, 70), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
            
            # Draw flashing border
            if frame_count % 6 < 3:  # Flash effect
                cv2.rectangle(annotated_frame, (5, 5), (frame_width-5, frame_height-5), 
                             (0, 0, 255), 8)
                
            # If we've captured an image, show confirmation
            if image_captured:
                capture_status = "Image Captured!"
                cv2.putText(annotated_frame, capture_status, (20, 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
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
        info_text = f"Frame: {frame_count} | Waste Objects: {len(objects_in_frame)} | People: {len(person_bboxes)}"
        cv2.putText(annotated_frame, info_text, (20, frame_height - 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Smart Waste Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

detect_waste_dumping()