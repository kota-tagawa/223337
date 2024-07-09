import cv2

img = cv2.imread("facemesh.png")
buf = img.copy()
cv2.drawMarker(buf,(1360,490),(255,0,0),markerType=cv2.MARKER_STAR,markerSize = 20,thickness=5)
cv2.drawMarker(buf,(2740,490),(255,0,0),markerType=cv2.MARKER_STAR,markerSize = 20,thickness=5)
cv2.drawMarker(buf,(1400,1640),(255,0,0),markerType=cv2.MARKER_STAR,markerSize = 20,thickness=5)
cv2.drawMarker(buf,(2700,1640),(255,0,0),markerType=cv2.MARKER_STAR,markerSize = 20,thickness=5)
cv2.imwrite('test.png',buf)