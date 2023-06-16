'''
This script tracks a mouses location within the 3 chambers of the Social Interaction Test (SIT).
Tracking is done by using background subtraction to discard static pixels, then each frame is 
thresholded and processed to isolate the moving mouse. Then the centroid of the mouse is determined
and recorded on an output CSV. 

To improve tracking accuracy, only moving objects in the selected ROI are tracked. This prevents 
tracking of unwanted objects such as an experimenter moving through the frame. 

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
from pointInside import pointInside
    


if __name__ == "__main__":
    # Define the layout for the paramter window
    layout_params = [
        [sg.Text('Select Video File: '), sg.FileBrowse('Browse', key='-path-')],
        [sg.Text('Number of Arenas: '), sg.InputText(default_text='1', key='-numArenas-')],
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
    last_location = None

    csvFile = open(f"centroid_{base_name}.csv", "w", newline='')
    csvWriter = csv.writer(csvFile)

    # Use frame to get ROI
    ret, frame = video.read()

    # Define ROI
    if not roiSelected:
        ROIs = selectROIs(frame, numROIs)
        roiSelected = True

    # Split arena ROI into 3 equal ROIs
    x, y, w, h = ROIs[0]
    third = w // 3
    left = (x, y, third, h)
    center = (x + third, y, third, h)
    right = (x + 2 * third, y, third, h)

    # Loop through each frame of the video
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Apply background subtractor to the frame 
        fgndMask = bgSubtractor.apply(frame)

        # Process image to denoise / smooth
        processed = imgProc(fgndMask)

        # Draw each rectangle in different colours
        cv.rectangle(frame, (left[0], left[1]), (left[0] + left[2], left[1] + left[3]), (0, 0, 255), 2)
        cv.putText(frame, 'Left', (left[0], y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv.rectangle(frame, (center[0], center[1]), (center[0] + center[2], center[1] + center[3]), (0, 255, 0), 2)
        cv.putText(frame, 'Center', (center[0], y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv.rectangle(frame, (right[0], right[1]), (right[0] + right[2], right[1] + right[3]), (255, 0, 0), 2)
        cv.putText(frame, 'Right', (right[0], y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Track the largest object in ROI
        point = centroid(processed[y:y+h, x:x+w])
        # Convert back to whole frame coordinates
        point = np.array(point)
        dim = np.array([x, y])
        point = tuple(point + dim)
        location = None

        if pointInside(point, left):
            location = 'Left'
        elif pointInside(point, center):
            location = 'Center'
        elif pointInside(point, right):
            location = 'Right'
        else:
            location = last_location

        last_location = location
      
        # Add the point to the csv
        csvWriter.writerow([location, int(point[0]), int(point[1])])
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


