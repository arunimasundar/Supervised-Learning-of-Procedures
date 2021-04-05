# Supervised Learning of Procedures from Tutorial Videos



------
## Introduction
This system produces a set of comprehensible, step-by-step instructions from video input. The videos can be of two categories: Academic Videos and Exercise Videos. To enable extracting procedures from these videos, Text Recognition, Speech Recognition and Action Recognition are used.


------
## Dependencies
 - python >= 3.5
 - Opencv >= 3.4.1   
 - scikit-learn==0.22.2
 - tensorflow & keras
 - numpy & scipy 
 - pathlib
 - pytesseract
 - SpeechRecognition
 
 
------
## Usage
 - Download the graph models from [here](https://drive.google.com/drive/folders/17Z7MwdQNTMPFvgms96S6XNdHV_nxATuy?usp=sharing), and place it under the Pose folder.
 - `python main.py e videoname.mp4`, will **perform action and speech recognition** and write the actions and speech recognized from the video to **exercise_video_output.txt**.
 - `python main.py e webcam`, will **perform action recognition** in realtime.
 - `python main.py a videoname.mp4`, will **perform text and speech recognition** and write the text and speech recognized from the video to **academic_video_output.txt**.




------
## Training with own dataset
 - prepare data(actions) by running `main.py`, remember to ***uncomment the code of data collecting***, the origin data will be saved as a `.txt`.
 - transforming the `.txt` to `.csv`, you can use EXCEL to do this.
 - do the training with the `train.py` in `Action/training/`, remember to ***change the action_enum and output-layer of model***.
 
 
------
## Test result
 - ***Check exercise_video_output.txt for exercise videos and academic_video_output.txt for academic videos.***



------
## Acknowledge
Thanks to the following works:    
 - [Online-Realtime-Action-Recognition-based-on-OpenPose](https://github.com/LZQthePlane/Online-Realtime-Action-Recognition-based-on-OpenPose)   
 