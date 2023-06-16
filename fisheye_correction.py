import cv2 as cv
import numpy as np
import glob
import os 


if __name__ == "__main__":

    # Code from opencv docs https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    
    # Termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('./calibration_images/*.jpg')
    i = 0
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (9,6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (7,6), corners2, ret)
            cv.imshow('img', img)
            cv.imwrite(f'drawn_chess/drawn_chess_{i}.jpg', img)
            cv.waitKey(500)

        i +=0
    cv.destroyAllWindows()

    # Undistort the videos: 
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Process each frame again for undistortion
    videos = glob.glob('/Volumes/Extreme SSD/Behavioural_pilot_videos/SIT/*.mp4')

    # Unidistort all .mp4 videos in the chosen folder
    for video_path in videos:
        # Create video capture object
        cap = cv.VideoCapture(video_path)

        # Get video parameters
        frameRate = cap.get(cv.CAP_PROP_FPS)
        # frameSize = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        frameSize = (1887, 1023)
        fourcc = cv.VideoWriter_fourcc(*'mp4v')

        # Get the base file name 
        file_name = os.path.basename(video_path)
        split = file_name.split('.')
        base_name = split[0]

        # Create output video object
        output = cv.VideoWriter(f"/Volumes/Extreme SSD/Behavioural_pilot_videos/calibrated_SIT/{base_name}.mp4", fourcc, frameRate, frameSize)



        # Loop through each frame of the video 
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Undistort the frame
            h, w = frame.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            if i == 0:
                print(type(dst))
                print(dst.size)
                print(dst.shape)
            cv.imshow('Undistorted Frame', dst)
            output.write(dst)
            i+=1
            if cv.waitKey(1) >= 0: 
                break

        cap.release()
        output.release()
        cv.destroyAllWindows()
        print("finished calibrating video")

    print("finished correcting all videos")