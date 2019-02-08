import cv2
import glob
import numpy as np
import imutils
import json
import sys

#from matplotlib import pyplot as plt


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



#images = glob.glob('bd_projet_scia/mires/*')
images = glob.glob('./*.jpg')
images = glob.glob('./*.JPG')
images += glob.glob('./*.png')
images += glob.glob('./*.PNG')
for name in sys.argv[1:]:
    try:
        images += glob.glob(name + '*.jpg')
        images += glob.glob(name + '*.JPG')
        images += glob.glob(name + '*.png')
        images += glob.glob(name + '*.PNG')
    except:
        print("ERROR wrong folder name")
        break

img_mire = cv2.imread('data/mire.png',0)    
img_damier = cv2.imread('data/damier.png',0)    


detected_or_not = {}
for fname in images:
        detected = False
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


        if feature_matching(resized, img_damier):
                detected = True
        if feature_matching(resized, img_mire):
                detected = True
        detected_or_not[fname] = detected


        with open('result_part1.json', 'w') as fp:
                json.dump(detected_or_not, fp)





