import cv2
import numpy as np

# Read the original image
img = cv2.imread('model.jpg') 
# Display original image
cv2.imshow('Original Image', img)
cv2.waitKey(0)

# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
# Display Sobel Edge Detection Images
cv2.imshow('Sobel X Edge Detection Image', sobelx)
#cv2.waitKey(0)
cv2.imshow('Sobel Y Edge Detection Image', sobely)
cv2.waitKey(0)
cv2.imshow('Sobel Edge Detection Image(Sobel X + Sobel Y', sobelxy)
cv2.waitKey(0)

#Prewitt Edge Detection
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]]) # Prewitt Edge Detection on the X axis
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]) # Prewitt Edge Detection on the Y axis
img_prewittx = cv2.filter2D(img_blur, -1, kernelx)
img_prewitty = cv2.filter2D(img_blur, -1, kernely)
# Display prewitt Edge Detection Image
cv2.imshow("Prewitt X Edge Detection Image", img_prewittx)
#cv2.waitKey(0)
cv2.imshow("Prewitt Y Edge Detection Image", img_prewitty)
cv2.waitKey(0)
cv2.imshow("Prewitt Edge Detection Image (Prewitt X + Prewitt Y)", img_prewittx + img_prewitty) # Combined X and Y Prewitt Edge Detection
cv2.waitKey(0)

#Roberts Edge Detection
kernelx = np.array([[1, 0], [0, -1]]) # Roberts Edge Detection on the X axis
kernely = np.array([[0, 1], [-1, 0]]) # Roberts Edge Detection on the X axis
img_robertx = cv2.filter2D(img_blur, -1, kernelx)
img_roberty = cv2.filter2D(img_blur, -1, kernely)
grad = cv2.addWeighted(img_robertx, 0.5, img_roberty, 0.5, 0)
#Display Roberts Edge Detection Image
cv2.imshow("Roberts X Edge Detection Image", img_robertx)
#cv2.waitKey(0)
cv2.imshow("Roberts Y Edge Detection Image", img_roberty)
cv2.waitKey(0)
cv2.imshow("Roberts Edge Detection Image (Robert X + Robert Y)", img_robertx + img_roberty) # Combined X and Y Roberts Edge Detection
cv2.waitKey(0)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) 
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

#for accessing live camera
cap = cv2.VideoCapture(0)

#for accessing vidoe file


while(1):
    # reads frames from a camera
    ret, frame = cap.read()

    cv2.imshow('Original Video',frame)
 
    # finds edges in the input image and
    # marks them in the output map edges
    edges = cv2.Canny(frame,100,100)
 
    cv2.imshow('Canny Edges Detction Vidoe',edges)

    laplacian = cv2.Laplacian(frame, cv2.CV_64F)#64f stands for 64 bit 
    laplacian = np.uint8(laplacian)
    cv2.imshow('Laplacian Edge Detction Video',laplacian)

    # Wait for Esc key to stop
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
 
 
# Close the window
cap.release()
cv2.destroyAllWindows()