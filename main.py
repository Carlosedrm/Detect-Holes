import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("Photos/buraco.jpg")
cv.imshow("Hole", img)

blur = cv.medianBlur(img, 31)
cv.imshow("Blur", blur)

gray = cv.cvtColor(blur, cv.COLOR_RGB2GRAY)
thresh, bin = cv.threshold(gray, 125, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
cv.imshow("Thresh", bin)  

canny = cv.Canny(bin, 75, 200)
cv.imshow('Canny', canny)

contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cv.drawContours(img, contours, -1, (100, 255, 0), 3)
cv.imshow('Contours', img)

cv.imwrite("arquivo_de_saida.jpg", img)

cv.waitKey(0)
