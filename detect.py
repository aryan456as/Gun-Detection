import cv2
from matplotlib import pyplot as plt

img = cv2.imread("image.jpg")


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

stop_data = cv2.CascadeClassifier('cascade.xml')

found = stop_data.detectMultiScale(img_gray, minSize =(425, 425))


amount_found = len(found) 
print("No of guns=", len(found))

if amount_found != 0: 
	
	
	for (x, y, width, height) in found: 
		
		# green rectangle around 
		# every recognized gun 
		cv2.rectangle(img_rgb, (x, y), 
					(x + height, y + width), 
					(0, 255, 0), 5) 
		

plt.subplot(1, 1, 1) 
plt.imshow(img_rgb) 
plt.show() 

