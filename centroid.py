import cv2 as cv


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