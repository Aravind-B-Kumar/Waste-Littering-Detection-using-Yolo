from ultralytics import YOLO
import cv2

# r"D:\MiniProject\dataset trash\QP.LPDE.v2i.yolov11\runs\detect\train3\weights\best.pt"
# r"D:\MiniProject\dataset trash\GarbageDetectionAPI.v1i.yolov11\runs\detect\train\weights\last.pt"
# r"D:\MiniProject\dataset trash\GarbageDetectionAPI.v1i.yolov11\runs\detect\train2\weights\last.pt"

# Load the trained model
model = YOLO(r"D:\MiniProject\dataset trash\GarbageDetectionAPI.v1i.yolov11\runs\detect\train6\weights\last.pt")#r"D:\MiniProject\dataset trash\QP.LPDE.v2i.yolov11\runs\detect\train3\weights\best.pt") #YOLO(r"D:\MiniProject\littering\Littering.v8-yolotiny.yolov11\runs\detect\train3\weights\last.pt")

# Open the webcam
cap = cv2.VideoCapture(1)#"http://192.168.137.138:8080/video")
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame, device=0, conf=0.5)  # Use GPU (device=0) and set confidence threshold

    # Filter out detections below 0.5 confidence (already handled by conf=0.5 in model inference)
    # Access detections if needed for further processing
    detections = results[0].boxes.data.cpu().numpy()  # Get detections as numpy array

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

  #Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()


#--------------------------------------------------------------------------

# import cv2
# from ultralytics import YOLO

# # Load the YOLOv8 model
# model = YOLO(r"D:\MiniProject\dataset trash\normal yolov8 garbage\runs\detect\train\weights\best.pt")  # Replace with your model path
# class_names = model.names 
# # Open the webcam
# cap = cv2.VideoCapture(1)  # 0 usually refers to the default webcam

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture image.")
#         break

#     results = model(frame, device=0, conf=0.5)

#     # Get detections and class names
#     detections = results[0].boxes.data.cpu().numpy()
#     class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Get class IDs as integers
#      # Get class names from the model

#     # Visualize and display class names
#     annotated_frame = results[0].plot()  # Draw bounding boxes and labels

#     # Iterate through detections and add class names manually (optional, but more control)
#     for i, det in enumerate(detections):
#         x1, y1, x2, y2, confidence = det[:5]  # Extract bounding box coordinates and confidence
#         class_id = class_ids[i]  # Get the class ID for the current detection
#         class_name = class_names[class_id]  # Get the class name using the ID
#         cv2.putText(annotated_frame, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Add class name to the frame

#     cv2.imshow("YOLOv8 Webcam", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()



#--------------------------------------------------------------

# working for detecting both garbage and person

# from ultralytics import YOLO
# import cv2

# # Load the trained models
# garbage_model = YOLO(r"D:\MiniProject\dataset trash\normal yolov8 garbage\runs\detect\train\weights\best.pt")
# person_model = YOLO(r"D:\MiniProject\dataset person\Persondetection\runs\detect\train4\weights\best.pt")  # Replace with the actual path to your person model

# # Open the webcam
# cap = cv2.VideoCapture(1)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture image.")
#         break

#     # Run YOLOv8 inference on the frame for both models
#     garbage_results = garbage_model(frame, device=0, conf=0.5)  # Use GPU (device=0) and set confidence threshold
#     person_results = person_model(frame, device=0, conf=0.5)    # Use GPU (device=0) and set confidence threshold

#     # Visualize the results from the garbage model
#     annotated_frame = garbage_results[0].plot()

#     # Visualize the results from the person model on the same frame
#     annotated_frame = person_results[0].plot(img=annotated_frame)

#     # Display the annotated frame
#     cv2.imshow("YOLOv8 Webcam", annotated_frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Release the webcam and close the window
# cap.release()
# cv2.destroyAllWindows()

#------------------------------------------------------------------


# from ultralytics import YOLO
# import cv2
# import numpy as np

# # Load the trained models
# garbage_model = YOLO(r"D:\MiniProject\dataset trash\normal yolov8 garbage\runs\detect\train\weights\best.pt")
# person_model = YOLO(r"D:\MiniProject\dataset person\Persondetection\runs\detect\train4\weights\best.pt")  # Replace with the actual path to your person model

# # Open the webcam
# cap = cv2.VideoCapture(1)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# # Distance threshold (in pixels)
# DISTANCE_THRESHOLD = 100  # Adjust this value based on your use case

# def calculate_distance(box1, box2):
#     """
#     Calculate the Euclidean distance between the centers of two bounding boxes.
#     Each box is in the format [x1, y1, x2, y2].
#     """
#     # Calculate center of box1
#     center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
#     # Calculate center of box2
#     center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
#     # Calculate Euclidean distance
#     distance = np.linalg.norm(center1 - center2)
#     return distance

# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture image.")
#         break

#     # Run YOLOv8 inference on the frame for both models
#     garbage_results = garbage_model(frame, device=0, conf=0.5)  # Use GPU (device=0) and set confidence threshold
#     person_results = person_model(frame, device=0, conf=0.5)    # Use GPU (device=0) and set confidence threshold

#     # Get garbage and person detections
#     garbage_boxes = garbage_results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, class]
#     person_boxes = person_results[0].boxes.data.cpu().numpy()    # [x1, y1, x2, y2, conf, class]

#     # Visualize the results from the garbage model
#     annotated_frame = garbage_results[0].plot()

#     # Visualize the results from the person model on the same frame
#     annotated_frame = person_results[0].plot(img=annotated_frame)

#     # Check distances between garbage and persons
#     for garbage_box in garbage_boxes:
#         for person_box in person_boxes:
#             distance = calculate_distance(garbage_box[:4], person_box[:4])  # Pass only [x1, y1, x2, y2]
#             if distance > DISTANCE_THRESHOLD:
#                 # Display an alert on the frame
#                 cv2.putText(annotated_frame, "Alert: Waste Dumped!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#                 break  # Break out of the inner loop if an alert is triggered

#     # Display the annotated frame
#     cv2.imshow("YOLOv8 Webcam", annotated_frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Release the webcam and close the window
# cap.release()
# cv2.destroyAllWindows()


#-----------------------------------------------------------

#NOTE code has set a line, and if object crosses that line, it alerts

# from ultralytics import YOLO
# import cv2

# # Load the trained models
# garbage_model = YOLO(r"D:\MiniProject\dataset trash\Littering.v4-video-extraction.yolov8\runs\detect\train\weights\best.pt")
# person_model = YOLO(r"D:\MiniProject\dataset person\Persondetection\runs\detect\train4\weights\best.pt")

# # Open the webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# # Define ground level (e.g., 90% of the frame height)
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# ground_level = int(frame_height * 0.9)

# # Initialize variables to track the object
# object_tracked = False
# object_bbox = None

# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture image.")
#         break

#     # Run YOLOv8 inference on the frame for both models
#     garbage_results = garbage_model(frame, device=0, conf=0.5)  # Use GPU (device=0) and set confidence threshold
#     person_results = person_model(frame, device=0, conf=0.5)    # Use GPU (device=0) and set confidence threshold

#     # Visualize the results from the garbage model
#     annotated_frame = garbage_results[0].plot()

#     # Visualize the results from the person model on the same frame
#     annotated_frame = person_results[0].plot(img=annotated_frame)

#     # Check if a waste object is detected
#     if len(garbage_results[0].boxes) > 0:
#         # Get the bounding box of the first detected waste object
#         object_bbox = garbage_results[0].boxes.xyxy[0].cpu().numpy()
#         x1, y1, x2, y2 = map(int, object_bbox)

#         # Check if the object is near or below the ground level
#         if y2 >= ground_level:
#             cv2.putText(annotated_frame, "Alert: Waste Thrown!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#             print("Alert: Waste Thrown!")

#         # Draw the ground level line
#         cv2.line(annotated_frame, (0, ground_level), (frame.shape[1], ground_level), (0, 255, 0), 2)

#     # Display the annotated frame
#     cv2.imshow("YOLOv8 Webcam", annotated_frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Release the webcam and close the window
# cap.release()
# cv2.destroyAllWindows()

#-------------------------------------

# from ultralytics import YOLO
# import cv2
# import numpy as np

# # Load the trained models
# garbage_model = YOLO(r"D:\MiniProject\dataset trash\trash kaggle\runs\detect\train4\weights\best.pt")
# person_model = YOLO(r"C:\Users\Admin\Downloads\best.pt") #"D:\MiniProject\dataset person\Persondetection\runs\detect\train4\weights\best.pt"
# throw_model = YOLO(r"C:\Users\Admin\Downloads\litter detetct.pt")  # Replace with the path to your throwing action model

# # Open the webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# # Distance threshold (in pixels)
# DISTANCE_THRESHOLD = 100  # Adjust this value based on your use case

# # Define ground level (e.g., 90% of the frame height)
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# ground_level = int(frame_height * 0.9)

# def calculate_distance(box1, box2):
#     """
#     Calculate the Euclidean distance between the centers of two bounding boxes.
#     Each box is in the format [x1, y1, x2, y2].
#     """
#     # Calculate center of box1
#     center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
#     # Calculate center of box2
#     center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
#     # Calculate Euclidean distance
#     distance = np.linalg.norm(center1 - center2)
#     return distance

# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture image.")
#         break

#     # Run YOLOv8 inference on the frame for all models
#     garbage_results = garbage_model(frame, device=0, conf=0.5)  # Use GPU (device=0) and set confidence threshold
#     person_results = person_model(frame, device=0, conf=0.5)    # Use GPU (device=0) and set confidence threshold
#     throw_results = throw_model(frame, device=0, conf=0.5)      # Use GPU (device=0) and set confidence threshold

#     # Get garbage, person, and throwing action detections
#     garbage_boxes = garbage_results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, class]
#     person_boxes = person_results[0].boxes.data.cpu().numpy()    # [x1, y1, x2, y2, conf, class]
#     throw_boxes = throw_results[0].boxes.data.cpu().numpy()      # [x1, y1, x2, y2, conf, class]

#     # Visualize the results from all models
#     annotated_frame = garbage_results[0].plot()
#     annotated_frame = person_results[0].plot(img=annotated_frame)
#     annotated_frame = throw_results[0].plot(img=annotated_frame)

#     # Check for throwing action and waste dumping
#     for person_box in person_boxes:
#         # Check if the person is performing the throwing action
#         for throw_box in throw_boxes:
#             if calculate_distance(person_box[:4], throw_box[:4]) < DISTANCE_THRESHOLD:
#                 # Check if waste is near the person and reaches the ground
#                 for garbage_box in garbage_boxes:
#                     distance = calculate_distance(person_box[:4], garbage_box[:4])
#                     if distance < DISTANCE_THRESHOLD and garbage_box[3] >= ground_level:  # Check if waste reaches the ground
#                         pass  # Add your logic here if needed

#     # Display the annotated frame
#     cv2.imshow("YOLOv8 Webcam", annotated_frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Release the webcam and close the window
# cap.release()
# cv2.destroyAllWindows()



##------------------------------------------------------------

# from ultralytics import YOLO
# import cv2
# import numpy as np
# from collections import defaultdict

# # Constants (Updated for better tracking)
# PROXIMITY_THRESHOLD = 100  #Pixel distance for interaction
# MIN_CARRY_FRAMES = 5       # Frames of continuous carry needed
# MIN_SIZE_CHANGE = 0.2      #  Size change thres  hold
# ALERT_DURATION = 30        #  Frames to show alert
# CONFIDENCE_THRESHOLD = 0.5 #  Detection confidence

# # Initialize models
# garbage_model = YOLO(r"D:\MiniProject\dataset trash\trash kaggle\runs\detect\train3\weights\best.pt")
# person_model = YOLO(r"C:\Users\Admin\Downloads\best.pt")

# cap = cv2.VideoCapture(0)

# # Improved tracking system
# class ObjectTracker:
#     def __init__(self):
#         self.next_id = 0
#         self.tracks = {}
    
#     def update(self, current_objects):
#         updated_tracks = {}
#         for obj in current_objects:
#             # Simple centroid matching (replace with proper tracking)
#             match_id = self._find_closest(obj['center'])
#             if match_id is not None:
#                 updated_tracks[match_id] = obj
#             else:
#                 updated_tracks[self.next_id] = obj
#                 self.next_id += 1
#         self.tracks = updated_tracks
    
#     def _find_closest(self, point):
#         for obj_id, obj in self.tracks.items():
#             if calculate_distance(obj['center'], point) < 50:
#                 return obj_id
#         return None

# person_tracker = ObjectTracker()
# garbage_tracker = ObjectTracker()

# def get_center(bbox):
#     x1, y1, x2, y2 = bbox
#     return ((x1 + x2) / 2, (y1 + y2) / 2)

# def get_size(bbox):
#     x1, y1, x2, y2 = bbox
#     return (x2 - x1) * (y2 - y1)

# def calculate_distance(p1, p2):
#     return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# track_history = defaultdict(lambda: {
#     'associated_garbage': None,
#     'carry_frames': 0,
#     'last_size': None,
#     'alert_counter': 0
# })

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     frame = cv2.resize(frame, (640, 480))
    
#     # Detect objects
#     garbage_results = garbage_model(frame, conf=CONFIDENCE_THRESHOLD)
#     person_results = person_model(frame, conf=0.7)
    
#     # Update trackers
#     current_garbage = [{'bbox': box.xyxy[0].cpu().numpy(), 
#                        'center': get_center(box.xyxy[0].cpu().numpy()),
#                        'size': get_size(box.xyxy[0].cpu().numpy())}
#                       for box in garbage_results[0].boxes]
    
#     current_persons = [{'bbox': box.xyxy[0].cpu().numpy(),
#                        'center': get_center(box.xyxy[0].cpu().numpy())}
#                       for box in person_results[0].boxes]
    
#     garbage_tracker.update(current_garbage)
#     person_tracker.update(current_persons)
    
#     # Process interactions
#     alerts = []
#     for p_id, person in person_tracker.tracks.items():
#         closest_garbage = None
#         min_dist = float('inf')
        
#         # Find closest garbage
#         for g_id, garbage in garbage_tracker.tracks.items():
#             dist = calculate_distance(person['center'], garbage['center'])
#             if dist < PROXIMITY_THRESHOLD and dist < min_dist:
#                 min_dist = dist
#                 closest_garbage = garbage
        
#         history = track_history[p_id]
        
#         if closest_garbage:
#             if history['associated_garbage'] == None:
#                 # New association
#                 history['associated_garbage'] = id(closest_garbage)
#                 history['last_size'] = closest_garbage['size']
#                 history['carry_frames'] = 1
#             else:
#                 # Check size consistency
#                 size_change = abs(closest_garbage['size'] - history['last_size']) / history['last_size']
#                 if size_change > MIN_SIZE_CHANGE:
#                     alerts.append(closest_garbage['bbox'])
#                     history['alert_counter'] = ALERT_DURATION
#                     history['carry_frames'] = 0
#                 else:
#                     history['carry_frames'] += 1
#                     history['last_size'] = closest_garbage['size']
#         else:
#             if history['carry_frames'] >= MIN_CARRY_FRAMES:
#                 history['alert_counter'] = ALERT_DURATION
#             history['carry_frames'] = 0
#             history['associated_garbage'] = None
    
#     # Draw results
#     annotated_frame = garbage_results[0].plot()
#     annotated_frame = person_results[0].plot(img=annotated_frame)
    
#     # Show alerts
#     alert_active = False
#     for p_id, history in track_history.items():
#         if history['alert_counter'] > 0:
#             alert_active = True
#             history['alert_counter'] -= 1
    
#     if alert_active:
#         cv2.putText(annotated_frame, "ILLEGAL DUMPING ALERT!", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
#     cv2.imshow("Waste Monitoring", annotated_frame)
    
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


