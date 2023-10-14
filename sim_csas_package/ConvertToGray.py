import cv2

image = cv2.imread('../sassed_images/staple.jpg')

gray_image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (200,200))

cv2.imwrite('../sassed_images/staple_gray.png', gray_image)