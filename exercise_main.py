# -*- coding: UTF-8 -*-
import cv2 as cv
import argparse
import numpy as np
import time
from utils import choose_run_mode, load_pretrain_model, set_video_writer
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize,output
from pathlib import Path

# file = open("output.txt","r+")
# file. truncate(0)
# file. close()
import tensorflow as tf
import speechtotext as sp
import sys


def exercise(video):

    

    
        


    # parser = argparse.ArgumentParser(description='Action Recognition by OpenPose')
    # parser.add_argument('--video', help='Path to video file.')
    # args = parser.parse_args()
    # print("args",args)
    # print(video)
    

    # 导入相关模型
    estimator = load_pretrain_model('VGG_origin')
    action_classifier = load_action_premodel('Action/sjphk_recognition.h5')

    # 参数初始化
    realtime_fps = '0.0000'
    start_time = time.time()
    fps_interval = 1
    fps_count = 0
    run_timer = 0
    frame_count = 0

    if video=='webcam':
        video= False
        cap=choose_run_mode(video)
        video_writer = set_video_writer(cap, write_fps=int(7.0))
        print("videowriter: ",video_writer)

    
    else:

        sp.stt(video)




        cap = choose_run_mode(video)
        # cap=cv.VideoCapture(video)
        print("cap: ",cap)
        video_writer = set_video_writer(cap, write_fps=int(7.0))
        print("videowriter: ",video_writer)

    file = open("action_output_befspeech.txt","r+")
    file. truncate(0)
    file. close()

    file1 = open("exercise_video_output.txt","r+")
    file1.truncate(0)
    file1.close()

    # f = open('highknees_data.txt', 'a+')

    while cv.waitKey(1) < 0:
        has_frame, show = cap.read()
        if has_frame:
            fps_count += 1
            frame_count += 1

            # pose estimation
            humans = estimator.inference(show)
            # get pose info
            pose = TfPoseVisualizer.draw_pose_rgb(show, humans)  # return frame, joints, bboxes, xcenter
            # recognize the action framewise
            show = framewise_recognize(pose, action_classifier)

            height, width = show.shape[:2]
            # FPS
            if (time.time() - start_time) > fps_interval:
                # interval，interval1，FPS
                realtime_fps = fps_count / (time.time() - start_time)
                fps_count = 0  
                start_time = time.time()
            fps_label = 'FPS:{0:.2f}'.format(realtime_fps)
            cv.putText(show, fps_label, (width-160, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            
            num_label = "Human: {0}".format(len(humans))
            cv.putText(show, num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            
            if frame_count == 1:
                run_timer = time.time()
            run_time = time.time() - run_timer
            time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
            cv.putText(show, time_frame_label, (5, height-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv.imshow('Action Recognition based on OpenPose', show)
            video_writer.write(show)

            
            # joints_norm_per_frame = np.array(pose[-1]).astype(np.str)
            # f.write(' '.join(joints_norm_per_frame))
            # f.write('\n')

    video_writer.release()
    cap.release()
    # f.close()

    output()

# exercise()