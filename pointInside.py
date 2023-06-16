def pointInside(point, rectangle):
    x, y, w, h = rectangle
    px, py = point

    if (x <= px <= x+w and y <= py <= y+h):
        return True
    else:
        return False