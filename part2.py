import cv2
import glob
import numpy as np
import imutils
import json

#import matplotlib
#matplotlib.use("TkAgg")
#from matplotlib import pyplot as plt


def corners(rect_detected):
        pass
def baricenter(rect_detected):
        pass
def mire_number(rect_detected):
        pass

def getSubImage(rect, src):
    # Get center, size, and angle from rect
    center, size, theta = rect
    # Convert to int 
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # Get rotation matrix for rectangle
    M = cv2.getRotationMatrix2D( center, theta, 1)
    # Perform rotation on src image
    dst = cv2.warpAffine(src, M, src.shape[:2])
    out = cv2.getRectSubPix(dst, size, center)
    return out

def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 50 :
                return True
            elif i==row1-1 and j==row2-1:
                return False


def feature_matching(img1, img2):
        MIN_MATCH_COUNT = 5
        #img_mire = cv2.imread(img_ref,0)          # queryImage
        #img2 = cv2.imread(img,0) # trainImage
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
                if m.distance < 0.7*n.distance:
                        good.append(m)

        if len(good)>MIN_MATCH_COUNT:
                return True
                # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                # matchesMask = mask.ravel().tolist()
                # h,w,d = img1.shape
                # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                # dst = cv2.perspectiveTransform(pts,M)
                # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
                
                # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                #    singlePointColor = None,
                #    matchesMask = matchesMask, # draw only inliers
                #    flags = 2)
                # img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
                #plt.imshow(img3, 'gray'),plt.show()

        else:
                print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
                return False



images = glob.glob('bd_projet_scia/mire/*')
img_mire = cv2.imread('data/mire.png',0)    
img_damier = cv2.imread('data/damier.png',0)    


result = {}
for fname in images:
        detected_list = []
        raw_image = cv2.imread(fname)


        scale_percent = 30 # percent of original size
        width = int(raw_image.shape[1] * scale_percent / 100)
        height = int(raw_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(raw_image, dim, interpolation = cv2.INTER_AREA)
        #cv2.imshow('a', resized)
        #cv2.waitKey(0)

        gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)

        bilateral_filtered_image = cv2.bilateralFilter(gray, 5, 175, 175)
        #cv2.imshow('a', bilateral_filtered_image)
        #cv2.waitKey(0)


        low_threshold = 75
        high_threshold = 200
        edges = cv2.Canny(bilateral_filtered_image, low_threshold, high_threshold)
        #cv2.imshow('a', edges)
        #cv2.waitKey(0)

        edges_copy = edges.copy()
        cnts = cv2.findContours(edges_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
                #cv2.imshow("a", copy)
                #cv2.waitKey(0)

                LENGTH = len(cnts)
                status = np.zeros((LENGTH,1))

                for i,cnt1 in enumerate(cnts):
                        x = i    
                        if i != LENGTH-1:
                                for j,cnt2 in enumerate(cnts[i+1:]):
                                        x = x+1
                                        dist = find_if_close(cnt1,cnt2)
                                        if dist == True:
                                                val = min(status[i],status[x])
                                                status[x] = status[i] = val
                                        else:
                                                if status[x]==status[i]:
                                                        status[x] = i+1

                unified = []
                maximum = int(status.max())+1
                for i in range(maximum):
                        pos = np.where(status==i)[0]
                        if pos.size != 0:
                                cont = np.vstack(cnts[i] for i in pos)
                                hull = cv2.convexHull(cont)
                                unified.append(hull)

                img_copy = resized.copy()
                cv2.drawContours(img_copy,unified,-1,(0,255,0),2)
                cv2.drawContours(edges_copy,unified,-1,255,-1)
                #cv2.imshow("a", img_copy)
                #cv2.waitKey(0)

                copy2 = resized.copy()
                for contour in unified:
                        rect = cv2.minAreaRect(contour)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        cv2.drawContours(copy2,[box],0,(0,0,255),2)
                        
                        rect_detected = getSubImage(rect, resized)
                        #cv2.imshow('a', rect_detected)
                        #cv2.waitKey(0)
                        

                        detection = {}
                        detected = False
                        if feature_matching(rect_detected, img_damier):
                                detection['type'] = 'damier'
                                detected = True
                        elif feature_matching(rect_detected, img_mire):
                                detection['type'] = 'mire'
                                detected = True
                                #detection['mire'] = mire_number(rect_detected)
                        if detected:
                                #detection['corner'] = corners(rect_detected)
                                #detection['center'] = baricenter(rect_detected)
                                detected_list.append(detection)
                        
                #cv2.imshow("a", copy2)
                #cv2.waitKey(0)
                
                
        result[fname] = detected_list


        with open('result_part2.json', 'w') as fp:
                json.dump(result, fp)





