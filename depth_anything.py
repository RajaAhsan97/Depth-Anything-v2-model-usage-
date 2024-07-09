"""
    Depth Anything v2 [https://github.com/DepthAnything/Depth-Anything-V2/tree/main]
    model output analyzing for different images and video input and also interface webcam.
    The pre-trained model is used by intergrating pipeline through transformers. I have used
    Small model which uses 24.8M parameters.

    By running the modle on my webcam, the model takes approximately 6 seconds to process
    the captured frame for estimating depth. And for videos, the resolution is scaled down
    to a factor of 4, which takes 2 seconds to process the frame.
"""

import os
import numpy as np
import cv2
from transformers import pipeline
from PIL import Image
import time

# Depth Anything model types
#   1.  Small [params{24.8M}]
#   2.  Base  [params{97.5M}]
#   3.  Large [params{335.3M}]
#   4.  Giant [params{1.3BM}]

# load pipeline
task = 'depth-estimation'
# url for model usage [https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf]
model = 'depth-anything/Depth-Anything-V2-Small-hf'
pipeline = pipeline(task=task, model=model)

# for interfacing webcam the argument of VideoCapture instance to 0
cap = cv2.VideoCapture('interior_design.mp4')

# get resolution of the frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)//4)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)//4)

out_frame = np.zeros((height, width*2, 3), np.uint8)

while True:
    # read frame captured from webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (width, height))    

    # convert image to pillow image object for processing with transformers pipeline
    image = Image.fromarray(frame)

    start = time.time()
    # infer depth 
    depth = pipeline(image)['depth']
    end = time.time()
    print(np.round(end-start, 2))

    # convert image back to np array hfor visualization
    depth_img = np.array(depth)
    depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_VIRIDIS)    

    # concatenate the frame and the estimated depth map 
    out_frame[:height, :width, :] = frame
    out_frame[:height, width:width*2, :] = depth_img

    # display the output
    cv2.imshow('frame', out_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
