import cv2
import numpy as np

# Example 1 to detect lines
# im = cv2.imread('address-tds.jpg')
# rows,cols = im.shape[:2]
# imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imgray,125,255,0)
# thresh = (255-thresh)
# thresh2=thresh.copy()
# im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# cv2.imshow('image1',im)
# cv2.imshow('image3',thresh2)
# cv2.drawContours(im, contours, -1, (0,255,0), 3) #draw all contours
# contnumber=4
# # cv2.drawContours(im, contours, contnumber, (0,255,0), 3) #draw only contour contnumber
# # cv2.imshow('contours', im)
# cv2.imwrite('countours.jpg', im)

# [vx,vy,x,y] = cv2.fitLine(contours[contnumber], cv2.DIST_L2,0,0.01,0.01)
# lefty = int((-x*vy/vx) + y)
# righty = int(((cols-x)*vy/vx)+y)
# cv2.line(im,(cols-1,righty),(0,lefty),(0,255,255),2)

# cv2.imshow('result', im)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# sample 2
# image = cv2.imread('address-tds.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 100, 250)
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, 25, minLineLength=100, maxLineGap=50)

# hough = np.zeros(image.shape, np.uint8)

# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(hough, (x1, y1), (x2, y2), (255, 255, 255), 2)

# cv2.imwrite('hough.jpg', hough)

gray = cv2.imread('address-tds.jpg')
edges = cv2.Canny(gray,50,150,apertureSize = 3)
cv2.imwrite('edges-50-150.jpg',edges)
minLineLength=1
lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=600,lines=np.array([]), minLineLength=minLineLength,maxLineGap=40)

a,b,c = lines.shape
for i in range(a):
    cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imwrite('hough.jpg',gray)