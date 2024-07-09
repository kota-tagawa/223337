import cv2

img = cv2.imread("facemesh.png")
buf = img.copy()
cv2.drawMarker(buf,(2050,440),(255,0,0),markerType=cv2.MARKER_STAR,markerSize = 20,thickness=5)
cv2.drawMarker(buf,(1150,1540),(255,0,0),markerType=cv2.MARKER_STAR,markerSize = 20,thickness=5)
cv2.drawMarker(buf,(2950,1540),(255,0,0),markerType=cv2.MARKER_STAR,markerSize = 20,thickness=5)
cv2.drawMarker(buf,(2050,1640),(255,0,0),markerType=cv2.MARKER_STAR,markerSize = 20,thickness=5)
cv2.drawMarker(buf,(990,600),(0,0,255),markerType=cv2.MARKER_STAR,markerSize = 20,thickness=5)
cv2.drawMarker(buf,(3100,600),(0,0,255),markerType=cv2.MARKER_STAR,markerSize = 20,thickness=5)
cv2.drawMarker(buf,(2050,1100),(0,0,255),markerType=cv2.MARKER_STAR,markerSize = 20,thickness=5)
cv2.drawMarker(buf,(2050,1280),(0,0,255),markerType=cv2.MARKER_STAR,markerSize = 20,thickness=5)
cv2.imwrite('test.png',buf)