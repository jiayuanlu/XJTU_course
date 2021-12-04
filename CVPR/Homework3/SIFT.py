import cv2

# import matplotlib.pyplot as plt

# %matplotlib inline

#reading image

img1 = cv2.imread('1.jpeg')

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

#keypoints

sift = cv2.xfeatures2d.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

img_1 = cv2.drawKeypoints(gray1,keypoints_1,img1)

cv2.imshow('img_1',img_1)
cv2.imwrite('Eiffel_feature1.jpeg',img_1)

img2 = cv2.imread('2.jpeg')

gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#keypoints

sift = cv2.xfeatures2d.SIFT_create()

keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

img_2 = cv2.drawKeypoints(gray2,keypoints_2,img2)

cv2.imshow('img_2',img_2)
cv2.imwrite('Eiffel_feature2.jpeg',img_2)


# # read images

# img1 = cv2.imread('eiffel_2.jpeg')

# img2 = cv2.imread('eiffel_1.jpg')

# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# figure, ax = plt.subplots(1, 2, figsize=(16, 8))

# ax[0].imshow(img1, cmap='gray')

# ax[1].imshow(img2, cmap='gray')


# #sift

# sift = cv2.xfeatures2d.SIFT_create()

# keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

# keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

# len(keypoints_1), len(keypoints_2)

# read images

# img1 = cv2.imread('2.jpeg')

# img2 = cv2.imread('1.jpeg')

# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# #sift

# sift = cv2.xfeatures2d.SIFT_create()

# keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

# keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

#feature matching

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)

matches =sorted(matches, key=lambda x:x.distance)

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)

cv2.imshow('img_3',img3)#,plt.show()
cv2.imwrite('Eiffel_match.jpeg',img3)
cv2.waitKey()
cv2.destroyAllWindows()