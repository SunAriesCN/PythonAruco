# Program to create custom ArUco dictionary using OpenCV and detect markers using webcam
# original code from: http://www.philipzucker.com/aruco-in-opencv/
# Modified by Iyad Aldaqre
# 12.07.2019

import numpy as np
import cv2
import cv2.aruco as aruco

# we will not use a built-in dictionary, but we could
# aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

# define an empty custom dictionary with 
# aruco_dict = aruco.custom_dictionary(0, 4, 1)
aruco_dict = aruco.Dictionary_create_from(1000, 5, aruco.getPredefinedDictionary( aruco.DICT_5X5_1000), 2)

# add empty bytesList array to fill with 3 markers later
print(aruco_dict.bytesList.shape)
aruco_dict.bytesList = np.append(aruco_dict.bytesList, np.empty(shape = (4, 4, 4), dtype = np.uint8), axis=0)

# add new marker(s)
mybits = np.array([[1,1,1,0,0],
                   [1,0,1,0,0],
                   [1,1,1,0,0],
                   [1,0,1,0,0],
                   [1,0,1,0,0]], dtype = np.uint8)
aruco_dict.bytesList[1000] = aruco.Dictionary_getByteListFromBits(mybits)

mybits = np.array([[0,1,1,0,0],
                   [1,0,0,1,0],
                   [1,1,1,0,0],
                   [1,0,0,1,0],
                   [0,1,1,0,0]],dtype = np.uint8)
aruco_dict.bytesList[1001] = aruco.Dictionary_getByteListFromBits(mybits)

mybits = np.array([[1,1,1,1,0],
                   [1,0,0,0,0],
                   [1,0,0,0,0],
                   [1,0,0,0,0],
                   [1,1,1,1,0]],dtype = np.uint8)
aruco_dict.bytesList[1002] = aruco.Dictionary_getByteListFromBits(mybits)

mybits = np.array([[0,1,1,1,0],
                   [1,0,0,0,0],
                   [1,1,1,0,0],
                   [1,0,0,0,0],
                   [1,1,1,1,0]],dtype = np.uint8)
aruco_dict.bytesList[1003] = aruco.Dictionary_getByteListFromBits(mybits)

fs = cv2.FileStorage("./logi_calib.yml", cv2.FILE_STORAGE_READ)
fn = fs.getNode("camera_matrix")
camera_matrix = np.array(fn.mat())
fn = fs.getNode("distortion_coefficients")
distorted_coeffs = np.array(fn.mat());

dictionary = cv2.FileStorage("./dictionary.yml", cv2.FILE_STORAGE_WRITE)
dictionary.write("bytesList", aruco_dict.bytesList)
dictionary.write("markerSize", 5)
dictionary.write("maxCorrectionBits", 2)

# # save marker images
# for i in range(len(aruco_dict.bytesList)):
#     cv2.imwrite("custom_aruco_" + str(i) + ".png", aruco.drawMarker(aruco_dict, i, 128))

# open video capture from (first) webcam
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        #lists of ids and the corners beloning to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict)
        # draw markers on farme
        frame = aruco.drawDetectedMarkers(frame, corners, ids)

        if len(corners) > 0:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 45, camera_matrix, distorted_coeffs)
            for rvec, tvec in zip(rvecs, tvecs):
                aruco.drawAxis(frame, camera_matrix, distorted_coeffs, rvec, tvec, 45)


        # resize frame to show even on smaller screens
        frame = cv2.resize(frame, None, fx = 0.8, fy = 0.8)
        # Display the resulting frame
        cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
