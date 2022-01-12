import cv2
import numpy as np
from PIL import Image

def order_rect(points):
    # idea: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    # initialize result -> rectangle coordinates (4 corners, 2 coordinates (x,y))
    res = np.zeros((4, 2), dtype=np.float32)    

    # top-left corner: smallest sum
    # top-right corner: smallest difference
    # bottom-right corner: largest sum
    # bottom-left corner: largest difference

    s = np.sum(points, axis = 1)    
    d = np.diff(points, axis = 1)

    res[0] = points[np.argmin(s)]
    res[1] = points[np.argmin(d)]
    res[2] = points[np.argmax(s)]
    res[3] = points[np.argmax(d)]

    return res

def four_point_transform(img, points):    
    # copied from: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_rect(points)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype = np.float32)

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
 
    # return the warped image
    return warped


def cont(img, gray, user_thresh, crop):
    found = False
    loop = False
    old_val = 0 # thresh value from 2 iterations ago
    i = 0 # number of iterations

    im_h, im_w = img.shape[:2]
    while found == False: # repeat to find the right threshold value for finding a rectangle
        if user_thresh >= 255 or user_thresh == 0 or loop: # maximum threshold value, minimum threshold value 
                                                 # or loop detected (alternating between 2 threshold values 
                                                 # without finding borders            
            break # stop if no borders could be detected

        ret, thresh = cv2.threshold(gray, user_thresh, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]        
        im_area = im_w * im_h
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (im_area/100) and area < (im_area/1.01):
                epsilon = 0.1 * cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,epsilon,True)

                if len(approx) == 4:
                    found = True
                elif len(approx) > 4:
                    user_thresh = user_thresh - 1
                    print(f"Adjust Threshold: {user_thresh}")
                    if user_thresh == old_val + 1:
                        loop = True
                    break
                elif len(approx) < 4:
                    user_thresh = user_thresh + 5
                    print(f"Adjust Threshold: {user_thresh}")
                    if user_thresh == old_val - 5:
                        loop = True
                    break

                rect = np.zeros((4, 2), dtype = np.float32)
                rect[0] = approx[0]
                rect[1] = approx[1]
                rect[2] = approx[2]
                rect[3] = approx[3]
                
                dst = four_point_transform(img, rect)
                dst_h, dst_w = dst.shape[:2]
                img = dst[crop:dst_h-crop, crop:dst_w-crop]
            else:
                if i > 100:
                    # if this happens a lot, increase the threshold, maybe it helps, otherwise just stop
                    user_thresh = user_thresh + 5
                    if user_thresh > 255:
                        break
                    print(f"Adjust Threshold: {user_thresh}")
                    print("WARNING: This seems to be an edge case. If the result isn't satisfying try lowering the threshold using -t")
                    if user_thresh == old_val - 5:
                        loop = True                
        i += 1
        if i%2 == 0:
            old_value = user_thresh

    return found, img


def invert(img):
    return ~img


def autocrop(PIL_img):
    thresh = 200
    crop = 0
    black_bg = True
    img = np.array(PIL_img)
    img = img[:, :, ::-1].copy() 

    if black_bg: # invert the image if the background is black
        img = invert(img)

    #add white background (in case one side is cropped right already, otherwise script would fail finding contours)
    img = cv2.copyMakeBorder(img,100,100,100,100, cv2.BORDER_CONSTANT,value=[255,255,255])
    im_h, im_w = img.shape[:2]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # res_gray = cv2.resize(img,(int(im_w/6), int(im_h/6)), interpolation = cv2.INTER_CUBIC)
    found, img = cont(img, gray, thresh, crop)

    if found:
        if black_bg:
            img = invert(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        return im_pil
    else:
        return PIL_img
