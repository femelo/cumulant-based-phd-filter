#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright: Flavio Eler De Melo
# E-mail: flavio.eler@gmail.com
import os
import argparse
import cv2
import numpy as np
import math
import sys
import pandas as pd
import copy as cp
import time
from termcolor import colored
from Cumulant_Filter.CumulantFilter import CumulantFilter
from Parameters import Parameters

# Global variables
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

if cv2.__version__[0] == '2':
    CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
    CAP_PROP_FPS = cv2.cv.CV_CAP_PROP_FPS
    AA = cv2.CV_AA
    FOURCC = cv2.cv.CV_FOURCC(*'mp4v')
elif cv2.__version__[0] >= '3':
    CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
    CAP_PROP_FPS = cv2.CAP_PROP_FPS
    AA = cv2.LINE_AA
    try:
        FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
    except:
        FOURCC = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

# Class for tracking application
class TrackingApplication(object):
    def __init__(self, parameters, display=True, show_messages=True, write_video=True):
        assert(isinstance(parameters, Parameters))
        self.parameters = parameters
        self.display = display
        self.showMessages = show_messages

        # Gather detections
        if parameters.tracking.det_file_ext[-3:] == 'csv':
            det_df = pd.read_csv(parameters.dets_file, sep=';')
            boxes = det_df.iloc[:,[0, 1, 2, 3, 4]].values
            # centroids = det_df.iloc[:,[0, 5, 6]].values
            frames_ids = list(set(boxes[:,0]))
            detections = dict()
            for i in range(len(frames_ids)):
                j = frames_ids[i]
                ind = np.where(boxes[:,0] == j)[0]
                corners = boxes[np.ix_(ind, [1, 2, 3, 4])].T
                pos_and_size = np.vstack((0.5*corners[[0, 1],:]+0.5*corners[[2, 3],:],corners[[2, 3],:]-corners[[0, 1],:]))
                detections[j] = pos_and_size
            self.detections = detections
        else:
            raise IOError('Detection file not supported.')

        self.frameCount = None
        self.frameRate = None
        self.videoWriter = None
        self.colorTable = None
        
        cap = cv2.VideoCapture(parameters.input_video_file)
        self.imgDim = (int(cap.get(3)), int(cap.get(4)))
        # Display tracking results?
        if self.display:
            self.cap = cap
            self.frameCount = int(self.cap.get(CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(CAP_PROP_FPS)
            if str(fps) == 'nan':
                self.frameRate = parameters.tracking.fps   # default to 30 if the metadata doesnt exist
            else:
                self.frameRate = fps
            self.frameTimebase = 1.0 / self.frameRate
            if write_video:
                self.videoWriter = cv2.VideoWriter(parameters.output_video_file, \
                    FOURCC, self.frameRate, self.imgDim)
            # Set colors to be used by tracks
            self.numOfColors = 1000  # 1000 different colors
            self.colorTable = np.random.choice(range(256), size=(self.numOfColors, 3))
        else:
            cap.release()

        if self.frameCount is None:
            self.numOfTimeSteps = max(self.detections.keys()) + 1
        else:
            self.numOfTimeSteps = min(self.frameCount, max(self.detections.keys()) + 1)
            if self.showMessages:
                print(colored('Processing and displaying {:d} frames with geometry {}x{} at {} frames/second.'.format(
                    self.numOfTimeSteps, self.imgDim[0], self.imgDim[1], self.frameRate), 'cyan', attrs=['bold']))

        # Instantiate multi-target tracker
        self.tracker = CumulantFilter(
            parameters=parameters.tracking,
            img_dim = self.imgDim,
            num_frames = self.numOfTimeSteps,
            debug=self.showMessages)

        # Set dataframe to keep tracking results
        self.outputFields = [
            'Frame ID', 'Track ID', 
            'Upper Left X', 'Upper Left Y', 
            'Lower Right X', 'Lower Right Y', 
            'Centroid X', 'Centroid Y', 
            'Centroid Velocity X', 'Centroid Velocity Y']
        self.logDataframe = pd.DataFrame(columns=self.outputFields)

    def draw_track(self, actual_img, box_coords, track_id, track_age):
        img = actual_img.copy()
        x, y, X, Y = map(int, box_coords)
        start_point = (x, y)
        end_point = (X, Y)
        color = tuple(map(int, self.colorTable[np.mod(track_id, self.numOfColors),:]))
        cv2.rectangle(img, start_point, end_point, color, 2)
        cv2.putText(img, '{},{}'.format(track_id, track_age), start_point, 
            FONT, .3, (255, 255, 255), 1, AA)
        return img

    def run(self):
        # Start counter and timer
        k = 0
        self.startTime = time.time()
        self.totalNumOfObjects = 0
        display_time = 0
        try:
            # Run loop
            while k < self.numOfTimeSteps:
                # Read new image
                if self.display:
                    start_time = time.time()
                    _, current_image = self.cap.read()
                    try:
                        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
                    except:
                        raise ValueError("Error converting to RGB.")
                    display_time += (time.time() - start_time)
                    if current_image is None:
                        k += 1
                        continue
                
                # Process detections
                if k in self.detections.keys():
                    # M = self.detections[k].shape[1]
                    observations = self.detections[k]
                else:
                    # M = 0
                    observations = np.array([[]])
                k += 1
                
                # One filter iteration
                self.tracker.iterate(observations)
                # Get updated estimates
                estimates_upd = self.tracker.estimates

                # Get estimates
                # estimates_keys = estimates_upd.keys()
                boxes, velocities, labels, tracks_ages, N, var_N, inact_status = \
                    estimates_upd['boxes'], estimates_upd['velocities'], estimates_upd['labels'], \
                    estimates_upd['times'], estimates_upd['N'], estimates_upd['var_N'], estimates_upd['inactive']
                num_of_objects = int(np.sum(self.tracker.intensity['w']))
                self.totalNumOfObjects += num_of_objects
                if self.showMessages:
                    print(colored(
                        'Frame: {0:03d}/{1:03d}, Number of tracks: {2:02d}, Expected number of objects: {3:02d}'.format(k, self.numOfTimeSteps, N, num_of_objects),
                        'cyan')
                    )

                for i in range(len(labels)):
                    x = boxes[0, i]
                    y = boxes[1, i]
                    w = boxes[2, i]
                    h = boxes[3, i]
                    X = x + w
                    Y = y + h
                    v_x = velocities[0, i]
                    v_y = velocities[1, i]
                    track_id = labels[i]
                    
                    if self.display:
                        start_time = time.time()
                        track_age = tracks_ages[i]
                        beta = inact_status[i]
                        alpha = 1.0 - beta
                        box_coords = (x, y, X, Y)
                        overlaid_image = self.draw_track(current_image, box_coords, track_id, track_age)
                        cv2.addWeighted(overlaid_image, alpha, current_image, beta, 0, current_image)
                        display_time += (time.time() - start_time)
                    # Save tracking output            
                    last_ind = self.logDataframe.shape[0]
                    self.logDataframe.loc[last_ind, self.outputFields] = np.array([k, track_id, x, y, X, Y, x + w/2.0, y + h/2.0, v_x, v_y])
                
                if self.display:
                    start_time = time.time()
                    # Add frame counter to the image
                    cv2.putText(current_image, 'Frame {:03d} / {:03d}, E[N] = {:03d} objects, N = {:03d} tracks'.format(
                        k, self.numOfTimeSteps, num_of_objects, N), (20, 20), \
                        FONT, 0.5, (255, 255, 255), 1, AA)
                    current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
                
                    if self.videoWriter is not None:
                        self.videoWriter.write(current_image)
                    # Render image
                    cv2.imshow('Tracking', current_image)
                    pressedKey = cv2.waitKey(25) & 0xFF
                    display_time += (time.time() - start_time)
                    if pressedKey == 27: # Wait for an ESC
                        print(colored('Tracking interrupted by the user (ESC).', 'yellow'))
                        break
        except KeyboardInterrupt:
            print(colored('\rTracking interrupted by the user (Ctrl-C).', 'yellow'))

        if k == self.numOfTimeSteps:
            print(colored('Tracking finished normally.', 'green', attrs=['bold']))
        
        if self.display:
            start_time = time.time()
            cv2.destroyAllWindows()
            self.cap.release()
            if self.videoWriter is not None:
                self.videoWriter.release()
            display_time += (time.time() - start_time)

        final_time = time.time()
        # Save output file
        self.logDataframe.to_csv(self.parameters.result_file, header=True, index=False)
        print(colored('Average number of objects per iteration = {:} objects/frame'.format(self.totalNumOfObjects / k), 'magenta'))
        print(colored('Average time per iteration = {:} s/frame'.format((final_time-self.startTime-display_time) / k), 'magenta'))

# Main function
def main():
    argparser = argparse.ArgumentParser(
        description='Tracking by the Cumulant-based PHD filter.')
    argparser.add_argument(
        '-ds', '--dataset',
        metavar='<dataset name>',
        default='WAMI',
        type=str,
        help='Dataset name (default: WAMI)')
    argparser.add_argument(
        '-iv', '--input-video',
        metavar='<input video name>',
        default='wami01',
        type=str,
        help='Input video name (default: wami01)')
    argparser.add_argument(
        '-s', '--settings-file',
        metavar='<settings file name>',
        default='settings.json',
        type=str,
        help='Settings file (default: \'settings.json\')')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='show_messages',
        help='Print information and debug messages')
    argparser.add_argument(
        '-d', '--display',
        action='store_true',
        dest='display',
        help='Display tracks during processing')
    argparser.add_argument(
        '-w', '--write-video',
        action='store_true',
        dest='write_video',
        help='Save video with the tracks presented')
    args = argparser.parse_args()

    # Check if settings file exists
    assert(os.path.isfile(args.settings_file))

    # Get parameters
    dataset_id = args.dataset
    video_name_components = args.input_video.split('.')
    if len(video_name_components) > 1:
        video_id = '.'.join(video_name_components[:-1])
    else:
        video_id = args.input_video

    # Instantiate the tracking application
    trackingApp = TrackingApplication(
        parameters = Parameters(args.settings_file, dataset_id, video_id), 
        display = args.display, 
        show_messages = args.show_messages, 
        write_video = args.write_video)

    # Run the application
    trackingApp.run()

if __name__ == "__main__":
    main()



