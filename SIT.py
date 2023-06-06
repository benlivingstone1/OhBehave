'''

- Ben Livingstone, June '23
'''

import cv2 as cv
import numpy as np
import PySimpleGUI as sg
import csv
import os
from imageProcessing import imgProc
from centroid import centroid
from selectROIs import selectROIs
    


if __name__ == "__main__":
    # Define the layout for the paramter window
    layout_params = [
        [sg.Text('Select Video File: '), sg.FileBrowse('Browse', key='-path-')],
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
    file = values['-path-']

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
    outputVid = cv.VideoWriter(f"tracked_{base_name}.mp4", fourcc, frameRate, frameSize)

    if not outputVid.isOpened():
        print("Error creating video writer")
        exit()

    # Create background subtractor object
    history = 2000
    varThreshold = 32.0
    bShadowDetection = True
    bgSubtractor = cv.createBackgroundSubtractorMOG2(history, varThreshold, bShadowDetection)

    # initialize required variables
    prevPoint = (0, 0)
    selectedRect = 0
    roiSelected = False
    numROIs = int(values['-numArenas-'])
    position_roi = []
    completed_position_roi = False

    csvFile = open(f"centroid_{base_name}.csv", "w", newline='')
    csvWriter = csv.writer(csvFile)

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


        # # Get ROI dimensions
        # x, y, w, h = ROIs[i]
        # # Draw the ROI
        # cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv.putText(frame, str(i + 1), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Track the largest object in ROI
        point = centroid(processed)

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

        # dimensions of the point
        px, py = point

        # Draw the ROIs
        # Get list of x value for all ROIs
        i = 0
        x_pos = []
        current_position = None
        for roi in ROIs:
            x, y, w, h = roi

            if (x <= px <= x+w and y <= py <= y+h):
                cv.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
                cv.putText(frame, position_roi[i], (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                current_position = position_roi[i]
            else:
                cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
                cv.putText(frame, position_roi[i], (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            i += 1

        if current_position == None:
            current_position = "CENTER"
      
        # Add the point to the csv
        csvWriter.writerow([current_position, int(point[0]), int(point[1])])
        # Draw the centroid of the tracked object on the frame
        cv.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

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


