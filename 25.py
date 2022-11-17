# ### Q2.5 Extra Credits: More 3D Visualization (10pts writeup) 

# Similar to the Q2.4 visualization of the Temple image, show the 3D reconstruction of any other object using your own images.
#  Feel free to use the images and camera parameters from the 
# [Middlebury multiview dataset](http://vision.middlebury.edu/mview/data/) 
# or other public datasets. 
# Please restrict to only using 2 images and atleast 10 correspondences in both the images.

# <span style='color:red'>**Output:**</span> In your write-up: 
# - Show the two images of the object from different views.
# - Visualize the 2D correspondences on both the images.
# - Take a few screenshots of the 3D visualization of the reconstruction.
import numpy as np
import cv2


# Visualization:
correspondence = np.load('data/some_corresp.npz') # Loading correspondences
intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
templeCoords = np.load("data/templeCoords.npz")
pts1, pts2 = correspondence['pts1'], correspondence['pts2']

im1, im2 = cv2.imread('images/dinoSR0001.png'),  cv2.imread('images/dinoSR0002.png')

# F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

# pts1 = np.hstack([templeCoords["x1"], templeCoords["y1"]])
# print(pts1.shape)
# P = compute3D_pts(pts1, intrinsics, F, im1, im2)

query_img = im1
train_img = im2
# plot_3D(P)
query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create()
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(queryDescriptors,trainDescriptors)
matches = sorted(matches, key = lambda x:x.distance)
# matches = matches[-3:]
final_img = cv2.drawMatches(query_img, queryKeypoints,
train_img, trainKeypoints, matches,None)
final_img = cv2.resize(final_img, (1000,650))
# Show the final imag
# print(matches)
for match in matches:
  p1 = queryKeypoints[match.queryIdx].pt
  p2 = trainKeypoints[match.trainIdx].pt
cv2.imshow("Matches", final_img)
cv2.waitKey(0)