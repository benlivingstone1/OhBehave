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


def imgProc(fgndMask):
    structuringElement = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    fgndMask = cv.GaussianBlur(fgndMask, (5, 5), 0)
    _, fgndMask = cv.threshold(fgndMask, 128, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    fgndMask = cv.morphologyEx(fgndMask, cv.MORPH_CLOSE, structuringElement, iterations=2)
    fgndMask = cv.morphologyEx(fgndMask, cv.MORPH_OPEN, structuringElement, iterations=2)

    return fgndMask


def centroid(fgndMask):
    # Find all the contours in the image
    contours, hierarchy = cv.findContours(fgndMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    contour_area = []
    for c in contours:
        contour_area.append(cv.contourArea(c))

    contour_area = sorted(contour_area, reverse=True)

    try:
        largest_contour = max(contours, key=cv.contourArea)
    except ValueError:
        largest_contour = None

    if largest_contour is not None:
        # Calculate the moments of the largest contour
        m = cv.moments(largest_contour)
        p = (m['m10'] / (m['m00'] + 1e-5), m['m01'] / (m['m00'] + 1e-5))

        return p
    else:
        p = (0,0)
        return p
    

def selectROIs(frame, numROIs):
    ROIs = []
    subArenas = {}
    for i in range(numROIs):
        rect = cv.selectROI(frame)
        ROIs.append(rect)

        x, y, w, h = rect
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(frame, str(i + 1), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return ROIs #, subArenas


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

        # For each ROI... 
        for i in range(len(ROIs)):
            # Get ROI dimensions
            x, y, w, h = ROIs[i]
            # Draw the ROI
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(frame, str(i + 1), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Track the largest object in ROI
            point = centroid(processed[y:y+h, x:x+w])
            # Convert back to whole frame coordinates
            point = np.array(point)
            dim = np.array([x, y])
            frame_point = tuple(point + dim)
            # Add the point to the csv
            csvWriter.writerow([i + 1, int(frame_point[0]), int(frame_point[1])])
            # Draw the centroid of the tracked object on the frame
            cv.circle(frame, (int(frame_point[0]), int(frame_point[1])), 5, (0, 255, 0), -1)

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


