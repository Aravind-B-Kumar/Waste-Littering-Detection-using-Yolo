import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

# Login Page
def login_page():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == "admin" and password == "123":
            st.session_state.logged_in = True
            st.success("Login successful! Redirecting...")
            st.rerun()  # Use st.rerun() to refresh the page
        else:
            st.error("Invalid username or password")

# Main Page with Waste Dumping Detection
def main_page():
    st.title("Waste Dumping Detection")
    
    # Button to start the webcam
    if st.button("Start Webcam", key="start_webcam"):
        # Initialize models
        model_pose = YOLO('yolo11l-pose.pt')
        model_object = YOLO(r"garbage_model.pt")
        
        # Open the webcam
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
            return
        
        # Placeholder for displaying the webcam feed
        frame_placeholder = st.empty()
        
        # Add a stop button outside the loop
        stop_button = st.button("Stop Webcam", key="stop_webcam")
        
        # Configuration
        HAND_OBJECT_DISTANCE = 150
        LITTER_CLASSES = ['garbage', 'rubbish']
        DROP_VELOCITY_THRESHOLD = 5
        history_length = 15
        DROP_COOLDOWN_FRAMES = 30
        
        # Tracking buffers
        hand_positions = defaultdict(list)
        object_positions = defaultdict(list)
        object_in_hand = defaultdict(bool)
        object_was_in_hand = defaultdict(bool)
        
        dropping_detected = False
        drop_cooldown = 0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Failed to capture image.")
                break
            
            frame_count += 1
            
            # Decrement cooldown counter if active
            if drop_cooldown > 0:
                drop_cooldown -= 1
                if drop_cooldown == 0:
                    dropping_detected = False
            
            # Run detection
            results_pose = model_pose.predict(frame)
            results_object = model_object.predict(frame, conf=0.25)
            
            # Create a copy of the frame for drawing
            annotated_frame = frame.copy()

            if results_pose[0].keypoints is not None:
                annotated_frame = results_pose[0].plot()
            
            # Draw pose keypoints and get hand positions
            current_hands = []
            if results_pose[0].keypoints is not None:
                for person_id, kps in enumerate(results_pose[0].keypoints.xy.cpu().numpy()):
                    if len(kps) >= 17:
                        left_wrist = kps[9].astype(float)
                        right_wrist = kps[10].astype(float)
                        current_hands.append(("left_" + str(person_id), left_wrist))
                        current_hands.append(("right_" + str(person_id), right_wrist))
                        
                        # Update hand position history
                        for hand_id, hand_pos in [("left_" + str(person_id), left_wrist), 
                                                 ("right_" + str(person_id), right_wrist)]:
                            hand_positions[hand_id].append(hand_pos)
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
                    if conf < 0.25:
                        continue
                    
                    class_idx = int(cls)
                    class_name = model_object.names[class_idx]
                    is_waste = any(waste_term in class_name.lower() for waste_term in LITTER_CLASSES)
                    
                    x1, y1, x2, y2 = box.astype(int)
                    obj_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                    obj_id = f"{class_name}_{idx}"
                    
                    if is_waste:
                        waste_detected = True
                        objects_in_frame.append(obj_id)
                        object_positions[obj_id].append(obj_center)
                        if len(object_positions[obj_id]) > history_length:
                            object_positions[obj_id].pop(0)
                        
                        near_hand = False
                        for hand_id, hand_pos in current_hands:
                            distance = np.linalg.norm(hand_pos - obj_center)
                            if distance < HAND_OBJECT_DISTANCE:
                                near_hand = True
                                object_in_hand[obj_id] = True
                                object_was_in_hand[obj_id] = True
                                break
                        
                        if object_was_in_hand[obj_id] and not near_hand:
                            if len(object_positions[obj_id]) >= 3:
                                recent_positions = object_positions[obj_id][-5:]
                                if len(recent_positions) >= 2:
                                    y_velocities = []
                                    for i in range(1, len(recent_positions)):
                                        y_velocity = recent_positions[i][1] - recent_positions[i-1][1]
                                        y_velocities.append(y_velocity)
                                    
                                    if sum(v > 0 for v in y_velocities) > len(y_velocities) / 2:
                                        avg_y_velocity = sum(y_velocities) / len(y_velocities)
                                        if avg_y_velocity >= DROP_VELOCITY_THRESHOLD:
                                            current_dropping_action = True
                                            dropping_detected = True
                                            drop_cooldown = DROP_COOLDOWN_FRAMES
                        
                        object_in_hand[obj_id] = near_hand
                        
                        if dropping_detected or current_dropping_action:
                            color = (0, 0, 255)
                            cv2.putText(annotated_frame, "DROPPING!", (x1, y1 - 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        elif near_hand:
                            color = (0, 255, 0)
                        else:
                            color = (255, 0, 0)
                        
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated_frame, f"{class_name}", (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        if len(object_positions[obj_id]) > 1:
                            for i in range(1, len(object_positions[obj_id])):
                                start = object_positions[obj_id][i-1].astype(int)
                                end = object_positions[obj_id][i].astype(int)
                                cv2.line(annotated_frame, tuple(start), tuple(end), (255, 0, 255), 1)
                    else:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                        cv2.putText(annotated_frame, f"{class_name}", (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Clean up objects not seen in this frame
            for obj_id in list(object_in_hand.keys()):
                if obj_id not in objects_in_frame:
                    object_in_hand[obj_id] = False
            
            # Display status
            if dropping_detected:
                status = "DROPPING WASTE DETECTED!"
                status_color = (0, 0, 255)
                cv2.putText(annotated_frame, status, (20, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
                if frame_count % 6 < 3:
                    cv2.rectangle(annotated_frame, (5, 5), (frame.shape[1]-5, frame.shape[0]-5), 
                                 (0, 0, 255), 8)
            elif waste_detected:
                status = "Waste Objects Detected"
                status_color = (255, 165, 0)
                cv2.putText(annotated_frame, status, (20, 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            else:
                status = "Monitoring..."
                status_color = (0, 255, 0)
                cv2.putText(annotated_frame, status, (20, 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # Show frame count and detection stats
            info_text = f"Frame: {frame_count} | Waste Objects: {len(objects_in_frame)}"
            cv2.putText(annotated_frame, info_text, (20, frame.shape[0] - 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Convert the annotated frame from BGR to RGB (Streamlit uses RGB)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display the annotated frame in the Streamlit app with a larger size
            frame_placeholder.image(annotated_frame_rgb, channels="RGB", width=1920)  # Adjust width as needed
            
            # Break the loop if the stop button is pressed
            if stop_button:
                break
        
        # Release the webcam
        cap.release()
        st.write("Webcam stopped.")

# Main App Logic
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        login_page()
    else:
        main_page()

if __name__ == "__main__":
    main()