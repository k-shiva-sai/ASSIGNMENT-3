#!/usr/bin/env python
# coding: utf-8

# In[4]:


import face_recognition as f
import numpy as np
import cv2


# In[5]:


shiva=f.load_image_file("C:\\Users\\Shiva\\Desktop\\LETS UPGRADE\\shiva.jpg")
shiva_encode=f.face_encodings(shiva)[0]
shiva_locate=f.face_locations(shiva)[0]

mahesh=f.load_image_file("C:\\Users\\Shiva\\Desktop\\LETS UPGRADE\\pichai.jpg")
mahesh_encode=f.face_encodings(mahesh)[0]
mahesh_locations=f.face_locations(mahesh)[0]

known_encode=[shiva_encode,mahesh_encode]
known_faces=["shiva","pichai"]

face_loc=[];face_encode=[];face_name=[]

cap=cv2.VideoCapture(0)
while True:
    flag,frame=cap.read()
    if not flag:
        print("NO LOADING")
        break
    frames=cv2.resize(frame,None,fx=0.25,fy=0.25)
    frame_rgb=cv2.cvtColor(frames,cv2.COLOR_BGR2RGB)
    frame_encode=f.face_encodings(frame_rgb)
    frame_location=f.face_locations(frame_rgb)
    face_name=[]
    for encode in frame_encode:
        match=f.compare_faces(known_encode,encode)
        name="unknown"
        face_dst=f.face_distance(known_encode,encode)
        idx=np.argmin(face_dst)
        if match[idx]:
            name=known_faces[idx]
        face_name.append(name)
        print(face_name)
    for (top,right,bottom,left),name in zip(frame_location,face_name):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame,(left,top),(bottom,right),(114,20,245))
        font=cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame,name,(left+10,bottom-10),font,0.75,(0,155,78),2)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1)& 0xff==ord('a'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




