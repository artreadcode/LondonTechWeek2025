import cv2

cv2.namedWindow("Reality?")
vc = cv2.VideoCapture(0) # Previously: 0 -> Logitech, 1 -> Computer

if vc.isOpened(): # Get the first frame
    rval, frame = vc.read()
else:
    rval = False
    
while rval:
    cv2.imshow("Reality?", frame)
    rval, frame = vc.read()
    
    if cv2.waitKey(1) & 0xFF == 27: # Press 'ESC' to exit
        break
    
vc.release()
cv2.destroyWindow("Reality?")