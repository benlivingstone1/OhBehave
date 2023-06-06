import cv2 as cv


def selectROIs(frame, numROIs):
    ROIs = []
    for i in range(numROIs):
        rect = cv.selectROI(frame)
        ROIs.append(rect)

        x, y, w, h = rect
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(frame, str(i + 1), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return ROIs