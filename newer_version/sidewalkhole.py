import cv2 as cv
import numpy as np

# Code
img = cv.imread("buracos.png")
cv.imshow("Hole", img)

# Transformations
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
cv.imshow("Gray", gray)

thresh, bin = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
cv.imshow("Thresh", bin)  

erosion = cv.erode(bin, (7,7) ,iterations = 1)
cv.imshow("erosion", erosion)

blur = cv.medianBlur(erosion, 13)
cv.imshow('Blur', blur)

canny = cv.Canny(blur, thresh, 200)
cv.imshow('Canny', canny)

contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# Grab's each countour.
con = []
for cnt in contours:
    area = cv.contourArea(cnt)
    con.append(area)

# Sorts the area of each contour and get's the smallest contour and the biggest contour.
con.sort()
smallest = con[0]
biggest = con.pop() - 1

for cnt in contours:
    area = cv.contourArea(cnt)
    # If there's more than one contour it will only draw the bigger one which is the hole. If there's only one contour it will draw the only countour.
    if len(contours) > 1:
        if smallest < area > biggest:
            cv.drawContours(img, [cnt], -1, (100, 255, 0), 2)
    else:
        cv.drawContours(img, [cnt], -1, (100, 255, 0), 2)

cv.imshow('Contours and final image', img)

cv.waitKey(0)