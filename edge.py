import cv2
import glob
import numpy as np
import imutils


images = glob.glob('bd_projet_scia/mires/*')

for fname in images:
        raw_image = cv2.imread(fname)


        scale_percent = 30 # percent of original size
        width = int(raw_image.shape[1] * scale_percent / 100)
        height = int(raw_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(raw_image, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow('a', resized)
        cv2.waitKey(0)

        gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)

        bilateral_filtered_image = cv2.bilateralFilter(gray, 5, 175, 175)
        cv2.imshow('a', bilateral_filtered_image)
        cv2.waitKey(0)


        low_threshold = 75
        high_threshold = 200
        edges = cv2.Canny(bilateral_filtered_image, low_threshold, high_threshold)
        cv2.imshow('a', edges)
        cv2.waitKey(0)

        cnts = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
        screenCnt = None

        # loop over our contours
        if len(cnts) > 0:
                for c in cnts:
                        # approximate the contour
                        peri = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
                
                        # if our approximated contour has four points, then
                        # we can assume that we have found our screen
                        if len(approx) == 4:
                                screenCnt = approx
                                break

                #cv2.drawContours(resized, [screenCnt], -1, (0, 255, 0), 3)
                copy = resized.copy()
                cv2.drawContours(copy, cnts, -1, (0,255,0), 3)
                cv2.imshow("a", copy)
                cv2.waitKey(0)

                for contour in cnts:
                        (x,y,w,h) = cv2.boundingRect(contour)
                        cv2.rectangle(resized, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.imshow("a", resized)
                cv2.waitKey(0)



        # rho = 1  # distance resolution in pixels of the Hough grid
        # theta = np.pi / 180  # angular resolution in radians of the Hough grid
        # threshold = 10  # minimum number of votes (intersections in Hough grid cell)
        # min_line_length = 40  # minimum number of pixels making up a line
        # max_line_gap = 20  # maximum gap in pixels between connectable line segments
        # line_image = np.copy(resized) * 0  # creating a blank to draw lines on

        # # Run Hough on edge detected image
        # # Output "lines" is an array containing endpoints of detected line segments
        # lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
        #                 min_line_length, max_line_gap)
        # if lines is not None:
        #         for line in lines:
        #                 for x1,y1,x2,y2 in line:
        #                         cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
        #         print(lines)
        # lines_edges = cv2.addWeighted(resized, 0.8, line_image, 1, 0)
        # cv2.imshow('a', line_image)
        # cv2.waitKey(0)
