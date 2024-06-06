import cv2

img = cv2.imread("facemesh.png")
buf = img.copy()
cv2.drawMarker(buf,(1150,1538),(255,0,0),markerType=cv2.MARKER_STAR,markerSize = 20,thickness=5)
cv2.drawMarker(buf,(1570,2845),(255,0,0),markerType=cv2.MARKER_STAR,markerSize = 20,thickness=5)
cv2.imwrite('test.png',buf)