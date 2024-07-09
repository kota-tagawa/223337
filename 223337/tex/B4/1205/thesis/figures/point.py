import cv2

img = cv2.imread("landmark.png")
buf = img.copy()
cv2.line(buf,(1150,1540),(2950,1540),(0,0,255),thickness=25)
cv2.drawMarker(buf,(1150,1540),(255,0,0),markerType=cv2.MARKER_STAR,markerSize = 20,thickness=50)
cv2.drawMarker(buf,(2950,1540),(255,0,0),markerType=cv2.MARKER_STAR,markerSize = 20,thickness=50)
cv2.drawMarker(buf,(2050,2245),(0,0,255),markerType=cv2.MARKER_STAR,markerSize = 20,thickness=50)
cv2.imwrite('landmark_world.png',buf)