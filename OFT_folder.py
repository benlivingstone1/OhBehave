'''
This script tracks the location of mice within an arena. Upon starting the script, the user will define the
number of arenas in the current video. The user will then be prompted to select the area corresponding to each 
arena. The script will then loop through each arena in each frame, and track the animal. The animals centroid
is saved as pixel coordinates in the output CSV and is drawn on the frame for the output video. 

- Ben Livingstone, June '23
'''

import cv2 as cv
import numpy as np
import PySimpleGUI as sg
import csv
import os
import glob
from imageProcessing import imgProc
from centroid import centroid
from selectROIs import selectROIs
from pointInside import pointInside
    

if __name__ == "__main__":
    # Define the layout for the paramter window
    layout_params = [
        [sg.Text('Select Video File: '), sg.FolderBrowse('Browse', key='-path-')],
        [sg.Text('Number of Arenas: '), sg.InputText(default_text='2', key='-numArenas-')],
        [sg.Button('Start')]
    ]

    # Create the paramter setting window
    window_params = sg.Window('Parameter Settings', layout_params)

    # Read events from window
    event, values = window_params.read()

    if event == sg.WINDOW_CLOSED:
        exit()
    
    # Declare video file you would like to track
    folder = values['-path-']

    # Create output folder if one doesn't exist
    output_csv = folder + '/output_csv/'
    output_mp4 = folder + '/output_videos/'

    if not os.path.exists(output_csv):
        os.mkdir(output_csv)
    if not os.path.exists(output_mp4):
        os.mkdir(output_mp4)

    # initialize required variables
    prevPoint = (0, 0)
    selectedRect = 0
    roiSelected = False
    numROIs = int(values['-numArenas-'])
    position_roi = []
    completed_position_roi = False
    ROIs = []

    for file in glob.glob(folder + "/*.mp4"):
        # Get the base file name
        base_file = os.path.basename(file)
        split_name = base_file.split('.')
        base_name = split_name[0]

        # Create a video capture object:
        # Put a filename as argument, otherwise "0" opens default camera
        video = cv.VideoCapture(file)

        # Check if the video is opened
        if not video.isOpened():
            "ERROR: Could not open input video."
            exit()

        # Get paramaters of input video 
        frameRate = video.get(cv.CAP_PROP_FPS)
        frameSize = (int(video.get(cv.CAP_PROP_FRAME_WIDTH)), int(video.get(cv.CAP_PROP_FRAME_HEIGHT)))

        # Create a VideoWriter object and use size from input video
        fourcc = cv.VideoWriter_fourcc(*'H264')
        outputVid = cv.VideoWriter(output_mp4 + f"tracked_{base_name}.mp4", fourcc, frameRate, frameSize)

        if not outputVid.isOpened():
            print("Error creating video writer")
            exit()

        # Create background subtractor object
        history = 2000
        varThreshold = 32.0
        bShadowDetection = True
        bgSubtractor = cv.createBackgroundSubtractorMOG2(history, varThreshold, bShadowDetection)

        csvFile = open(output_csv + f"centroid_{base_name}.csv", "w", newline='')
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(["rectangle", "location", "x", "y"])

        # Loop through each frame of the video
        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Apply background subtractor to the frame 
            fgndMask = bgSubtractor.apply(frame)

            # Process image to denoise / smooth
            processed = imgProc(fgndMask)

            # Define ROI
            if not roiSelected:
                ROIs = selectROIs(frame, numROIs)
                roiSelected = True

            if not completed_position_roi:
                # Determine spatial relationship between 2 ROIs
                if len(ROIs) == 2:
                    # Determine which ROI is on the left
                    difference = ROIs[0][0] - ROIs[1][0]
                    # Append location corresponding to same index as roi
                    if difference < 0:
                        position_roi.append('LEFT')
                        position_roi.append('RIGHT')
                    else:
                        position_roi.append('RIGHT')
                        position_roi.append('LEFT')
                # If there is more than 2 ROI's sort them from left to right (ascending values of x)
                else:
                    for i in range(len(ROIs)):
                        position_roi.append(i + 1)
                    position_roi.sort()

            # For each ROI... 
            for i in range(len(ROIs)):
                # Get ROI dimensions
                x, y, w, h = ROIs[i]
                # Draw the ROI
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv.putText(frame, position_roi[i], (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Draw middle zone in the ROI
                # dimensions of the center rectangle
                w_c = int(w * 2/3)
                h_c = int(h * 2/3)
                # cooridinates to start drawing the rectangle
                x_c = x + (w - w_c) // 2
                y_c = y + (h - h_c) // 2
                center_rect = (x_c, y_c, w_c, h_c)
                cv.rectangle(frame, (x_c, y_c), (x_c + w_c, y_c + h_c), (0, 0, 255), 2)

                # Track the largest object in ROI
                point = centroid(processed[y:y+h, x:x+w])
                # Convert back to whole frame coordinates
                point = np.array(point)
                dim = np.array([x, y])
                frame_point = tuple(point + dim)

                # Check if point is in the center
                if pointInside(frame_point, center_rect):
                    cv.circle(frame, (int(frame_point[0]), int(frame_point[1])), 5, (0, 0, 255), -1)
                    location = "center"
                else:
                    cv.circle(frame, (int(frame_point[0]), int(frame_point[1])), 5, (0, 255, 0), -1)
                    location = "edge"

                # Add the point to the csv
                csvWriter.writerow([position_roi[i], location, int(frame_point[0]), int(frame_point[1])])
                # Draw the centroid of the tracked object on the frame
                

            cv.imshow("frame", frame)

            outputVid.write(frame)

            if cv.waitKey(1) >= 0:
                break

        # Clean up
        video.release()
        outputVid.release()
        cv.destroyAllWindows()
        csvFile.close()

        print("finished tracking video")

    print("finished tracking all videos in folder!")


