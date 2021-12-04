import numpy as np
import cv2

left_img = cv2.imdecode(np.fromfile('18.jpg', dtype=np.uint8), 1)
left_img = cv2.resize(left_img, (600, 400))
right_img = cv2.imdecode(np.fromfile('17.jpg', dtype=np.uint8), 1)
right_img = cv2.resize(right_img, (600, 400))
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

hessian = 300
SIFT = cv2.xfeatures2d.SIFT_create(hessian) 
# surf = cv2.xfeatures2d.SIFT_create()
keypoint1, descriptor1 = SIFT.detectAndCompute(left_gray, None)  
keypoint2, descriptor2 = SIFT.detectAndCompute(right_gray, None)


drawKeyPoint_left = cv2.drawKeypoints(left_gray, keypoint1, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("img_with_drawKeyPoint_left", drawKeyPoint_left)
cv2.imwrite('left_SIFT.jpeg',drawKeyPoint_left)
drawKeyPoint_right = cv2.drawKeypoints(right_gray, keypoint2, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("img_with_drawKeyPoint_right", drawKeyPoint_right)
cv2.imwrite('right_SIFT.jpeg',drawKeyPoint_right)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(descriptor1,descriptor2)
matches =sorted(matches, key=lambda x:x.distance)
img3 = cv2.drawMatches(left_img, keypoint1, right_img, keypoint2, matches[:50], right_img, flags=2)

cv2.imshow('img_3',img3)
cv2.imwrite('match.jpeg',img3)

flann_index=0
index=dict(algorithm=flann_index,trees=5)
recursion=dict(iter=50)
flann_match=cv2.FlannBasedMatcher(index,recursion)

key_match=flann_match.knnMatch(descriptor1,descriptor2,k=2)

feature_point=[]
for i ,j in key_match:
    if i.distance<0.7*j.distance:
        feature_point.append(i)

src_descript=np.array([keypoint1[i.queryIdx].pt for i in feature_point])
train_descript=np.array([keypoint2[i.trainIdx].pt for i in feature_point])

H=cv2.findHomography(src_descript,train_descript,cv2.RANSAC,5)

high1,wide1=left_gray.shape[:2]
high2,wide2=right_gray.shape[:2]
trans=np.array([[1.0, 0, wide1], [0, 1.0, 0], [0, 0, 1.0]])

touying=np.dot(trans,H[0])

train=cv2.warpPerspective(left_img,touying,(wide1+wide2,max(high1,high2)))

cv2.imshow('left_img', train) 
cv2.imwrite('left.jpeg',train)
train[0:high2, wide1:wide1+wide2] = right_img  
cv2.imshow('total_img', train)
cv2.imwrite('merge.jpg',train)
cv2.imshow('leftgray', left_img)
cv2.imshow('rightgray', right_img)
cv2.waitKey(0)
cv2.destroyAllWindows()