import streamlit as st
from db.database import Database
from time import sleep
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import os
from datetime import datetime

db = Database()

def login_page():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if db.fetchOne('SELECT 1 FROM credentials WHERE username=? AND password=?',username,password):
            st.session_state.logged_in = True
            st.success("Login successful! Redirecting...")
            st.session_state.page = "main"  # Set to main page
            st.rerun()
        else:
            st.error("Invalid username or password")
    
    if st.button("Don't have an account? Sign up"):
        st.session_state.page = "signup"  # Switch to signup page
        st.rerun()


def signup_page():
    st.title("Sign Up Page")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    
    if st.button("Sign Up"):
        if db.fetchOne('SELECT 1 FROM credentials WHERE username=?',username):
            st.error("Username already exists. Please choose a different one.")
        else:
            db.execute("INSERT INTO credentials(username,password) VALUES(?,?)",username,password)
            st.success("Sign up successful! You can now log in.")
            st.session_state.page = "login"     
            sleep(1)
            st.rerun()

    if st.button("Already have an account? Log in"):
        st.session_state.page = "login"  
        st.rerun()


def view_captured_images():
    st.title("Captured Waste Dumping Images")
    
    captures_dir = "./captures"
    
    if not os.path.exists(captures_dir):
        st.warning("No captures directory found.")
        return
    
    image_files = [f for f in os.listdir(captures_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        st.info("No captured images found.")
        return
    
    images_by_date = {}
    for filename in image_files:
        try:
            # Handle both filename formats:
            # 1. waste_dropper_person0_28-03-25__06.57am.jpg
            # 2. waste_dropper_person0_20240328_065700.jpg
            
            if "__" in filename:  # Old format with double underscore
                # Split on last underscore before date
                date_part = filename.rsplit("_", 2)[-2]
                # Extract just the date portion (28-03-25)
                date_str = date_part.split("__")[0]
                date_obj = datetime.strptime(date_str, "%d-%m-%y")
            else:  # New format
                date_str = filename.split("_")[-2]  # Gets YYYYMMDD
                date_obj = datetime.strptime(date_str, "%Y%m%d")
            
            formatted_date = date_obj.strftime("%B %d, %Y")
            if formatted_date not in images_by_date:
                images_by_date[formatted_date] = []
            
            images_by_date[formatted_date].append(filename)
        except Exception as e:
            st.warning(f"Could not process filename {filename}: {str(e)}")
            continue
    
    # Sort dates in descending order
    sorted_dates = sorted(images_by_date.keys(), 
                         key=lambda x: datetime.strptime(x, "%B %d, %Y"), 
                         reverse=True)
    
    # Display images grouped by date
    for date in sorted_dates:
        st.subheader(date)
        date_images = images_by_date[date]
        cols = st.columns(3)
        
        for i, filename in enumerate(date_images):
            col = cols[i % 3]
            full_path = os.path.join(captures_dir, filename)
            
            with col:
                try:
                    st.image(full_path, caption=filename)
                    with open(full_path, "rb") as file:
                        st.download_button(
                            label="Download image",
                            data=file,
                            file_name=filename,
                            mime="image/jpeg"
                        )
                except Exception as e:
                    st.error(f"Failed to display {filename}: {str(e)}")


def main_page():
    st.title("Waste Dumping Detection")
    
    # Navigation buttons in the sidebar
    st.sidebar.title("Navigation")
    
    # Use buttons instead of radio
    if st.sidebar.button("Detect Waste"):
        st.session_state.current_nav = "Detect Waste"
    
    if st.sidebar.button("View Captured Images"):
        st.session_state.current_nav = "View Captured Images"
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"  # Redirect to login page
        st.rerun()
    
    # Determine which page to show based on navigation state
    if not hasattr(st.session_state, 'current_nav'):
        st.session_state.current_nav = "Detect Waste"
    
    if st.session_state.current_nav == "View Captured Images":
        view_captured_images()
    else:
        # Button to start the webcam
        if st.button("Start Webcam", key="start_webcam"):
            # Initialize models
            model_pose = YOLO(r'models/yolo11l-pose.pt')
            model_object = YOLO(r"models/last.pt")
            
            
            # Open the webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Error: Could not open webcam.")
                return
            
            # Set camera resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
            # Create captures directory if it doesn't exist
            captures_dir = "./captures"
            if not os.path.exists(captures_dir):
                try:
                    os.makedirs(captures_dir)
                    st.write(f"Created directory: {captures_dir}")
                except Exception as e:
                    st.error(f"Failed to create directory {captures_dir}: {e}")
                    # Fall back to current directory if we can't create /captures
                    captures_dir = "./captures"
                    if not os.path.exists(captures_dir):
                        os.makedirs(captures_dir)
            
            # Placeholder for displaying the webcam feed
            frame_placeholder = st.empty()
            
            # Add a stop button outside the loop
            stop_button = st.button("Stop Webcam", key="stop_webcam")
            
            # Configuration
            HAND_OBJECT_DISTANCE = 150
            LITTER_CLASSES = ['garbage', 'paper', 'plastic']
            DROP_VELOCITY_THRESHOLD = 5
            history_length = 15
            DROP_COOLDOWN_FRAMES = 30
            
            # Tracking buffers
            person_hand_objects = {}
            hand_positions = defaultdict(list)
            object_positions = defaultdict(list)
            object_hand_associations = {}
            person_bboxes = {}
            
            dropping_detected = False
            dropping_person_id = None
            dropping_object_id = None
            drop_cooldown = 0
            
            image_captured = False
            frame_count = 0
            
            # Skeleton Connections (COCO keypoint order)
            SKELETON_CONNECTIONS = [
                # Head and Neck
                (0, 1),   # Nose to Left Eye
                (0, 2),   # Nose to Right Eye
                (1, 3),   # Left Eye to Left Ear
                (2, 4),   # Right Eye to Right Ear
                
                # Upper Body
                (5, 7),   # Left Shoulder to Left Elbow
                (7, 9),   # Left Elbow to Left Wrist
                (6, 8),   # Right Shoulder to Right Elbow
                (8, 10),  # Right Elbow to Right Wrist
                
                # Torso and Spine
                (5, 6),   # Between Shoulders
                (5, 11),  # Left Shoulder to Left Hip
                (6, 12),  # Right Shoulder to Right Hip
                (11, 12), # Hip Line
                
                # Lower Body
                (11, 13), # Left Hip to Left Knee
                (13, 15), # Left Knee to Left Ankle
                (12, 14), # Right Hip to Right Knee
                (14, 16)  # Right Knee to Right Ankle
            ]
            
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
                        dropping_person_id = None
                        dropping_object_id = None
                        image_captured = False
                
                # Run detection
                results_pose = model_pose.predict(frame)
                results_object = model_object.predict(frame, conf=0.5)
                
                # Create a copy of the frame for drawing
                annotated_frame = frame.copy()
                
                # Track keypoints and process people
                current_hands = []
                person_bboxes = {}
                
                if results_pose[0].keypoints is not None:
                    # Get person bounding boxes
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
                            
                            # Draw skeleton connections
# Inside the pose detection loop where you draw skeleton connections:
                            for connection in SKELETON_CONNECTIONS:
                                if connection[0] < len(kps) and connection[1] < len(kps):
                                    start = kps[connection[0]].astype(int)
                                    end = kps[connection[1]].astype(int)
                                    
                                    # Skip if either point is invalid (0,0 or NaN)
                                    if (start[0] > 0 and start[1] > 0 and 
                                        end[0] > 0 and end[1] > 0 and
                                        not np.isnan(start).any() and not np.isnan(end).any()):
                                        
                                        # Different colors for different body regions
                                        if connection[0] in [0, 1, 2, 3, 4]:  # Head region
                                            color = (255, 0, 0)  # Blue
                                        elif connection[0] in [5, 6, 7, 8, 9, 10]:  # Upper body
                                            color = (0, 255, 0)  # Green
                                        else:  # Lower body
                                            color = (0, 0, 255)  # Red
                                        
                                        cv2.line(annotated_frame, tuple(start), tuple(end), color, 2)
                            
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
                                                timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
                                                
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
                                                        st.write(f"Saved image of person {dropping_person_id} to {filename}")
                                                        # Also save full frame for context
                                                        context_filename = f"{captures_dir}/waste_drop_context_{timestamp}.jpg"
                                                        cv2.imwrite(context_filename, frame)
                                                        image_captured = True
                                                    except Exception as e:
                                                        st.error(f"Failed to save image: {e}")
                        
                        # Draw bounding box and label for waste items
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated_frame, f"{class_name}", (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Draw object trajectory if tracked
# Inside the object detection loop where you draw trajectories:
                        if len(object_positions[obj_id]) > 1:
                            for i in range(1, len(object_positions[obj_id])):
                                start = object_positions[obj_id][i-1].astype(int)
                                end = object_positions[obj_id][i].astype(int)
                                
                                # Skip if either point is invalid (0,0 or NaN)
                                if (start[0] > 0 and start[1] > 0 and 
                                    end[0] > 0 and end[1] > 0 and
                                    not np.isnan(start).any() and not np.isnan(end).any()):
                                    
                                    cv2.line(annotated_frame, tuple(start), tuple(end), (255, 0, 255), 1)
                
                # Clean up object associations for objects not in this frame
# Inside the cleanup section for object associations:
                        for obj_id in list(object_hand_associations.keys()):
                            if obj_id not in objects_in_frame:
                                # Remove if the object is no longer detected
                                if obj_id in object_positions:
                                    del object_positions[obj_id]
                                if obj_id in object_hand_associations:
                                    del object_hand_associations[obj_id]
                        
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
                
                # Convert frame for Streamlit display
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
                
                # Break the loop if stop button is pressed
                if stop_button:
                    break
            
            # Release the webcam
            cap.release()
            st.write("Webcam stopped.")


def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "page" not in st.session_state:
        st.session_state.page = "login"  # Default to login page

    if st.session_state.logged_in:
        main_page()  # Show main application logic
    else:
        if st.session_state.page == "login":
            login_page()
        else:
            signup_page()


if __name__ == "__main__":
    main()