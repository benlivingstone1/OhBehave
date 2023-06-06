import cv2 as cv


def imgProc(fgndMask):
    structuringElement = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    fgndMask = cv.GaussianBlur(fgndMask, (5, 5), 0)
    _, fgndMask = cv.threshold(fgndMask, 128, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    fgndMask = cv.morphologyEx(fgndMask, cv.MORPH_CLOSE, structuringElement, iterations=2)
    fgndMask = cv.morphologyEx(fgndMask, cv.MORPH_OPEN, structuringElement, iterations=2)

    return fgndMask