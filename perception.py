#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ultralytics
from ultralytics import YOLO
import math
import matplotlib.pyplot as plt
import rotpy
import rotpy
import cv2
import numpy as np
from rotpy.system import SpinSystem
from rotpy.camera import CameraList
import matplotlib.pyplot as plt
import cv2
import numpy as np
from IPython.display import display, Image
import ipywidgets as widgets
import threading


# In[2]:


model = YOLO("best.pt")  # load an official model


# In[3]:



import pyflycap2 as pyfc


# In[4]:


import os


# In[5]:


import cv2
import math
import robotpy_apriltag  # Import from pupil-apriltags

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # adjust width
cap.set(4, 480)  # adjust height

# Initialize the AprilTag detector from pupil-apriltags
#os.add_dll_directory("C:/Users/zakol/anaconda3/Newfolder/lib/site-packages/pupil_apriltags/lib")
#"C:/Users/username/Miniconda3/envs/my_env/lib/site-packages/pupil_apriltags.libs"

#at_detector = Detector()

# Assuming your model is already loaded and initialized
# model = torch.load('your_model.pt') 
# model.eval()
system = SpinSystem()
cameras = CameraList.create_from_system(system, True, True)
#cameras.get_size()
camera = cameras.create_camera_by_serial('15374755')
camera.init_cam()
# the names of the pixel formats available for the camera
#camera.camera_nodes.PixelFormat.get_entries_names()
classNames = ['ball']  # Replace with your actual class names
camera.camera_nodes.PixelFormat.set_node_value_from_str('RGB8')  # or 'Y16' for 16-bit grayscale

#camera.end_acquisition()

while True:
    
    #success, img = cap.read()
    
    camera.begin_acquisition()
    image_cam = camera.get_next_image()
    img = np.array(image_cam.get_image_data())
    image_cam.release()

    camera.end_acquisition()
    img = img.reshape(964, 1288, 3)
    #img = img[:, :, ::-1]
    #if not success:
    #   break
    print(img.shape)
    # ---- 1. Run inference with your trained model ----
    results = model(img, stream=True)
    print(results)
    # Process the model's object detection results

    # ---- 2. AprilTag detection using pupil-apriltags ----
    # Convert the frame to grayscale for AprilTag detection
    detector = robotpy_apriltag.AprilTagDetector()
    detector.addFamily("tag25h9", 3)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    base_x = 0.0
    base_y = 0.0
    len_side = 1.0
    tags = detector.detect(gray_image)
    print(len(tags))
    # Draw bounding boxes around detected AprilTags
    tags.sort(key = lambda x: x.getId(), reverse = False)
    for tag in tags:
        if(tag.getId() == 0):
            side_len_pix = ((tag.getCorner(0).x - tag.getCorner(1).x) ** 2 + (tag.getCorner(0).y - tag.getCorner(1).y) ** 2) ** (1/2)
            len_side = -1 * 7.75 / side_len_pix
            base_x = tag.getCenter().x
            base_y = tag.getCenter().y
            #cv2.putText(img,  str(side_len_pix) , (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100),thickness = 1)
            #cv2.putText(img,  str(round(len_side, 4)), (500, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100),thickness = 1)
        labx = round((tag.getCenter().x - base_x) * len_side, 2)
        laby = round((tag.getCenter().y - base_y) * len_side, 2)
        #labx2 = round(tag.getCenter().x - base_x, 2)
        #laby2 = round(tag.getCenter().y - base_y, 2)
        labt = '(' + str(labx) + ", " + str(laby) + ')'
        #labt2 = '(' + str(labx2) + ", " + str(laby2) + ')'
        #print("-----------", labx, laby, labx2, laby2)
        #print(type(tag.getCenter().x))
        #print(tag.getId())
        #print("base_x, tag.getCenter().x,str(tag.getCenter().x - base_x), len_side, str((tag.getCenter().x - base_x)* len_side)")
        #print(base_x, tag.getCenter().x,str(tag.getCenter().x - base_x), len_side, str((tag.getCenter().x - base_x)* len_side))
        
        # Convert to integer coordinates
        for i in range(4):
            j = (i + 1) % 4
            point1 = (int(tag.getCorner(i).x), int(tag.getCorner(i).y))
            point2 = (int(tag.getCorner(j).x), int(tag.getCorner(j).y))
            cv2.line(img, point1, point2, (255, 0, 255), 2)

        cx = int(tag.getCenter().x)
        cy = int(tag.getCenter().y)
        ll = 10
        cv2.putText(img, str(tag.getId()), (cx + ll, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0),thickness = 2)
        cv2.putText(img,  labt , (cx + 12, cy + 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0),thickness = 2)
        #cv2.putText(img,  labt2 , (cx + 12, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 100),thickness = 1)



        # Display the tag ID at the center of the tag
#         tag_center = (int(tag.center[0]), int(tag.center[1]))
#         cv2.putText(img, str(tag.tag_id), tag_center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int
            midx = (x1 + x2)/2.0
            midy = (y1 + y2)/2.0
            # Draw the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Get the confidence score and class name
            confidence = math.ceil((box.conf[0] * 100)) / 100
            #print("Confidence --->", confidence)

            cls = int(box.cls[0])
            #print("Class name -->", classNames[cls])
            labt = '(' + str(round((midx - base_x)* len_side, 2)) + ", " + str(round((midy - base_y)* len_side, 2)) + ')'
            # Display class name on the bounding box
            org = [x1, y1]
            org2 = [x2, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 255, 0)
            thickness = 2
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            cv2.putText(img, labt, org2, font, fontScale, color, thickness)
    # ---- 3. Display the combined detections ----
    cv2.imshow("Webcam with Model and AprilTag Detection", img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
camera.deinit_cam()
camera.release()


# In[35]:


camera.deinit_cam()
camera.release()

