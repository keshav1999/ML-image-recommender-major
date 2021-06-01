import cv2
import numpy as np
import os

def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
        refPoint = [(x_start, y_start), (x_end, y_end)]
        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped", roi)
            save_link = os.path.join("model_images",dirs[product_id],image_list[image_id])
            if dirs[product_id] not in os.listdir('model_images'):
                os.mkdir(os.path.join('model_images',dirs[product_id]))
            cv2.imwrite(save_link,roi)
dirs = os.listdir('images')
if "model_images" not in os.listdir():
    os.mkdir("model_images")
product_id=0
image_id = 0
flag=True
while True:
    if flag == False:
        break
    image_list = os.listdir(os.path.join('images',dirs[product_id]))
    if len(image_list)==0:
        product_id+=1
        continue
    img_link = os.path.join('images',dirs[product_id],image_list[image_id])
    cropping = False
    x_start, y_start, x_end, y_end = 0, 0, 0, 0
    image = cv2.imread(img_link)
    oriImage = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)
    while True:
        i = image.copy()
        if not cropping:
            cv2.imshow("image", image)
        elif cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", i)
        a = cv2.waitKey(1)
        if a==ord('e'):
            cv2.destroyAllWindows()
            flag = False
            break
        if a==ord('a'):
            if image_id==0:
                cv2.destroyAllWindows()
                break
            else:
                image_id=image_id-1
                cv2.destroyAllWindows()
                break
        if a==ord('d'):
            if image_id==len(image_list)-1:
                cv2.destroyAllWindows()
                break
            else:
                image_id=image_id+1
                cv2.destroyAllWindows()
                break
        if a==ord('s'):
            cv2.destroyAllWindows()
            break
        if a==ord('r'):
            save_link = os.path.join("model_images",dirs[product_id],image_list[image_id])
            os.remove(save_link)
            cv2.destroyAllWindows()
            break
        if a==ord('n'):
            if image_id==len(dirs)-1:
                cv2.destroyAllWindows()
                break
            else:
                image_id=0
                product_id+=1
                cv2.destroyAllWindows()
                break
        if a==ord('p'):
            if image_id==0:
                cv2.destroyAllWindows()
                break
            else:
                image_id=0
                product_id-=1
                cv2.destroyAllWindows()
                break
        
                
        