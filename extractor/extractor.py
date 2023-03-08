import cv2
import numpy as np
from extractor.consts import *
from extractor.config import *
from extractor.stats import *

# ==== вспомогательные инструменты ====
 
# поиск крайних координат указателя
def find_xy_line(image_edited, cx, cy):
    contours, _ = cv2.findContours(image_edited,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    if len(contours) < 1:
        print(STATUS_error_distance)
        return -1,-1

    max_cntr_index = 0


    def min_Xy(cntr):
        x = 9999
        y = 0
        for p in cntr:
            if x > p[0][0]:
                x = p[0][0]
                y = p[0][1]
        return x,y

    def max_Xy(cntr):
        x = 0
        y = 0
        for p in cntr:
            if x < p[0][0]:
                x = p[0][0]
                y = p[0][1]
        return x,y
    
    def min_Yx(cntr):
        y = 9999
        x = 0
        for p in cntr:
            if y > p[0][1]:
                y = p[0][1]
                x = p[0][0]
        return y, x

    def max_Yx(cntr):
        y = 0
        x = 0
        for p in cntr:
            if y < p[0][1]:
                y = p[0][1]
                x = p[0][0]
        return y, x
    
    # относительно цетра ищем сектор (левый или правый)
    # где находится указатель
    cntr = contours[max_cntr_index]

    minxx = min_Xy(cntr)[0]
    maxyy, maxyx = max_Yx(cntr)[0], max_Yx(cntr)[1]
    minyy =  min_Yx(cntr)[0]

    if minxx < cx:
        if maxyy >= cy and minyy <= cy:
            return maxyx,maxyy
        if maxyy <= cy and minyy < cy:
            return min_Xy(cntr)

    if minxx >= cx:
        if minyy < cy:
            return min_Yx(cntr)[1],minyy
        if minyy == cy:
            return max_Xy(cntr)
        if maxyy > cy:
            return maxyx,maxyy

    return -1,-1

# поиск крайних координат контура надписи "TIM"
def find_xy_tim(max_countors):
    max_x = 99999
    min_x = 0
    x = 0
    for i in max_countors:
        if i[0][0] < max_x:
            max_x = i[0][0]
        if i[0][0] > min_x:
            min_x = i[0][0]
    x = (max_x - min_x)/2+min_x
    return int(x)

# перевести угол в градусы на приборе и добавить смещение от текстовой метки к нулю
def transform_angle(x):
    return int(x/CONST_D1_D2-CONST_BIAS_TIM_TO_ZERO)

# ==== основные функции ====

# получение значения
# filename = "file-image.jpg" // только jpg-файл
# modeview = True/False // режим визуализации вкл/выкл
def extract_value(filename, modeview):
    # проверка существования файла
    try:
        f = open(filename)
        f.close()
    except FileNotFoundError:
        print(STATUS_not_found_image)
        return
    angle = -1
    img = cv2.imread(filename)
    original_image = img
    
    w = int(img.shape[1] / CONST_RESIZE) 
    h = int(img.shape[0] / CONST_RESIZE)

    step_blur = 35

    for i in range(step_blur):
        if i % 2 == 1:
            continue

        img = cv2.imread(filename)
        original_image = img
        
        i = step_blur - i
        k_blur = i
        k_median = i+4

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.blur(img, (k_blur, k_blur))
        img = cv2.medianBlur (img, k_median)

        img = cv2.resize(img, (w,h))
        original_image = cv2.resize(original_image, (w,h))

        img = cv2.Canny(img, 40, 200, False, 3)

        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, img.shape[0]/64, 
                                param1=150, param2=50, 
                                minRadius=40, maxRadius=10000)


        if circles is not None:
            for cn in circles:
                for circle_n in cn:
                    c = circle_n
                    x,y,r = int(c[0]), int(c[1]), int(c[2])
                    cv2.circle(original_image, (x,y), r, USER_COLOR_LINE, 5)
                    x,y,r = int(circle_n[0]), int(circle_n[1]), int(circle_n[2])
                    c = (x,y,r)
                    
                    # распознование указателя, метки  и угла
                    x1,y1,_,_ = find_line_pointer(original_image, c)
                    if x1 == -1 or y1 == -1:
                        print(STATUS_not_found_line)
                        return
                    
                    xt,yt = find_tim_label(original_image, c)
                    if xt == -1 or yt == -1:
                        print(STATUS_not_found_angle)
                        return
                    
                    angle = transform_angle(find_angle(x1,y1,xt,yt,c[0],c[1]))
                    
                    # если распознан ложный - отрицательный угол
                    if int(angle) < 0:
                        print(STATUS_not_found_angle)
                        return
                    
                    if modeview:
                        # добавляем на изображение центр и указатель с подписью градусов
                        # и отображаем окно
                        cv2.circle(original_image, (c[0],c[1]), c[2], USER_COLOR_LINE, 5)
                        cv2.line(original_image, (x1,y1),(c[0],c[1]), USER_COLOR_LINE, thickness=USER_BOLD_LINE) 
                        font                   = cv2.FONT_HERSHEY_SIMPLEX
                        bottomLeftCornerOfText = (c[0]+5,c[1]+5)
                        fontScale              = USER_FONT_SIZE
                        fontColor              = USER_COLOR_LINE
                        thickness              = USER_BOLD_LINE
                        lineType               = 1
                        cv2.putText(original_image,str(angle),
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
                        cv2.imshow(filename, original_image)
                        cv2.waitKey(0)
                    return    str(angle)     
        else:
            continue    
    return str(angle)

# получить координаты указателя и центра
def find_line_pointer(image_origin_resize, center):
    img  = image_origin_resize.copy()
    cx, cy, cr = int(center[0]), int(center[1]), int(center[2])
    # two layer
    cv2.circle(img, (cx,cy), int(cr-cr/3.5), (255,255,255), int(cr/1.1))
    #  center
    cv2.circle(img, (cx,cy), int(cr/11), (255,255,255), int(cr/6))
    # out layer
    cv2.circle(img, (cx,cy), int(cr*3), (255,255,255), int(cr*4.2)) 

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img, (2,2))
    img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)[1]
    img = cv2.Canny(img, 50, 200, False, 3)

    x_line, y_line = find_xy_line(img, cx, cy)

    return x_line, y_line, cx, cy

# получить координаты надписи tim
def find_tim_label(image_origin_resize, center):
    img  = image_origin_resize.copy()
    cx, cy, cr = int(center[0]), int(center[1]), int(center[2])
    #  center
    cv2.circle(img, (cx,cy), int(cr/11), (255,255,255), int(cr/6))
    # out layer
    cv2.circle(img, (cx,cy), int(cr), (255,255,255), int(cr)) 
    cv2.circle(img, (cx,cy), int(cr*3), (255,255,255), int(cr*4.2)) 
    cv2.rectangle(img,(0,0),(int(cx*2), int(cy+cr/4)), (255,255,255), -1)
    cv2.rectangle(img,(0,cy),(int(cx*2), int(cy+cr/3)), (255,255,255), -1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img, (1,1))
    img = cv2.medianBlur (img, 7)
    img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)[1]
    img = cv2.Canny(img, 50, 200, False, 3)


    cntst, _ = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    if len(cntst) < 1:
        return -1, -1
    
    max_c = 0
    index_c = 0
    for i in range(len(cntst)):
        c = cntst[i]
        if len(c) > max_c:
            max_c = len(c)
            index_c = i

    max_cntr = cntst[index_c]
    xct = find_xy_tim(max_cntr)
    
    return xct, cy*2

# получить угол по двум отрезкам и цетру
def find_angle(x1,y1, x2,y2, c1,c2):
    a1 = np.array([x1,y1])
    c = np.array([c1,c2])
    a2 = np.array([x2,y2])
    a1c = a1 - c
    a2c = a2 - c
    cosine_angle = np.dot(a2c,a1c) / (np.linalg.norm(a1c) * np.linalg.norm(a2c))
    angle = np.degrees(np.arccos(cosine_angle))
    return int(angle)
