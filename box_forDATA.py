###################♥
##@作者：小小浪漫雨    ♥
##@声明：严禁转载        ♥
##@注意：参数条件见文档说明 ♥
##########################♥

import os
import cv2
import numpy as np
import pickle
import math
import glob


###############置信度设置和调用模型####################
confidence_threshold = 0.70
with open(r'D:\Ubuntu\ABC_DATA\knn_for_data_char\knn_data.pkl', 'rb') as f:
    knn = pickle.load(f)

#######################图像处理函数###################
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    equ = cv2.equalizeHist(binary)
    resized = cv2.resize(equ, (20, 20))
    flattened = resized.reshape(1, -1).astype(np.float32)
    return flattened

#######################图像预测函数###################
def recognize_letter(img_path):
    flattened = preprocess_image(img_path)
    confidences = knn.predict_proba(flattened)
    max_confidence = np.max(confidences)
    pred_label = knn.predict(flattened)[0]
    if max_confidence < confidence_threshold:
        return None
    else:
        return pred_label

#################数字定位函数#########################
def crop_image(img, x_offset, y_offset):
    half_width, half_height = 100, 100
    x1, y1 = x_offset - half_width, y_offset - half_height
    x2, y2 = x_offset + half_width, y_offset - half_height
    x3, y3 = x_offset + half_width, y_offset + half_height
    x4, y4 = x_offset - half_width, y_offset + half_height
    mask = np.zeros_like(img[:, :, 0])
    roi_corners = np.array([[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, (255, 255, 255))
    masked_image = cv2.bitwise_and(img, img, mask=mask)
    cropped_image = masked_image[y1:y3, x1:x3]
    return cropped_image

#############重叠面积计算函数###################
def get_overlap_area(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    area = w * h
    return area

###############中心坐标计算函数###################
def make_square_contour(rect):
    x, y, w, h = rect
    cx, cy = x + w // 2, y + h // 2
    size = max(w, h)
    x = cx - size // 2
    y = cy - size // 2
    return x, y, size, size

##############中心坐标##########
def get_box_center(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    return x1 + w // 2, y1 + h // 2

#########按x轴遍历轮廓#################
def sort_contours_x_axis(contours):
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contours, bounding_boxes = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0]))
    return contours

#############读取电脑自带摄像头#######
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_rect = None
    max_area = 10000
    for cnt in contours:
        # 面积判定
        area = cv2.contourArea(cnt)
        if area < max_area:
            continue
        # 矩形性判定
        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # 相对位置判定
        if w > 0.9 * h and x > img.shape[1] * 0.1 and y > img.shape[0] * 0.1 and x + w < img.shape[1] * 0.9 and y + h < img.shape[0] * 0.9:
            # 边缘密度判定
            edge = img[y:y + h, x:x + w]
            edge_gray = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)
            _, edge_thresh = cv2.threshold(edge_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            edge_count = cv2.countNonZero(edge_thresh)
            edge_density = edge_count / float(w * h)
            if edge_density > 0.5:
                max_rect = (x, y, w, h)
                max_area = area

    # 绘制最大的闭合黑色矩形框
    if max_rect is not None:
        x_min, y_min, width, height = max_rect
        x_max = x_min + width
        y_max = y_min + height
        cv2.rectangle(img, (max_rect[0], max_rect[1]), (max_rect[0]+max_rect[2], max_rect[1]+max_rect[3]), (0, 0, 255), 2)
        cropped = img[y_min:y_max, x_min:x_max]
        dim = (600, 400)
        resized = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite('box_data.jpg', resized)
        img = cv2.imread('box_data.jpg')


        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.erode(thresh, kernel)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        index = 0
        boxes = []
        centers = []
        contours = sort_contours_x_axis(contours)
        # 遍历轮廓
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            # 通过判断轮廓宽度和高度，过滤掉非数字区域
            if w > 50 and h > 50 and w < 150 and h < 150:
                new_center_x, new_center_y = get_box_center(x, y, x+w, y+h)  # 计算新的矩形框中心坐标
                overlap = False
                flag_data = True
                for box in boxes:
                    cx, cy = get_box_center(box[0], box[1], box[0] + box[2], box[1] + box[3])
                    centers.append((cx, cy))
                    for center in centers:
                        dist = math.sqrt((center[0] - new_center_x) ** 2 + (center[1] - new_center_y) ** 2)
                        if dist < 30:
                            # 判断是否与已经识别出的数字区域重叠
                            area1 = w * h
                            area2 = get_overlap_area((x, y, w, h), box)
                            if area2 / area1 >= 0.1:
                                overlap = True
                                break

                    # 额外的宽高比判定，针对数字1
                    aspect_ratio = float(w) / h
                    if h < 150 and ((aspect_ratio > 4 and w < 50) or (h > w and w < 40)):
                        flag_data = True  # 将宽高比不符合要求的轮廓过滤掉
                    if not flag_data:
                        boxes.append((x, y, w, h))
                        index += 1
                if not overlap:
                    # # 在原图上绘制数字矩形框
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
                    boxes.append((x, y, x + w, y + h))  # 将当前数字的左上角和右下角坐标加入到 boxs 列表中
                    roi = thresh[y:y + h, x:x + w]
                    roi = 255 * np.ones((h, w), dtype=np.uint8)
                    roi[0:h, 0:w] = thresh[y:y + h, x:x + w]
                    roi = 255 - roi
                    roi_resized = cv2.resize(roi, (40, 40), interpolation=cv2.INTER_AREA)
                    # cv2.imwrite('digit_{}.jpg'.format(index), roi_resized)
                    cv2.imwrite(f'C:\\Users\\aifei\\Desktop\\Robot_sofa_file\\knn_for_data_char\\digit_data\\digit_{index}.jpg', roi_resized)
                    index += 1

        # # 显示并保存绘制矩形框后的图像
        # cv2.imshow('resul22t', img)
        # cv2.imwrite('resul22t.jpg', img)

        # 获取所有符合条件的文件路径
        file_paths = glob.glob(r'C:\Users\aifei\Desktop\Robot_sofa_file\knn_for_data_char\digit_data\digit_*.jpg')
        predicted_str = ''
        # 遍历文件路径列表，依次处理每张图片
        for file_path in file_paths:
            predicted_letter = recognize_letter(file_path)
            predicted_letter_str = str(predicted_letter)
            predicted_str += predicted_letter_str
            # os.remove(file_path)

        print(f"预测结果为：{predicted_str}")

    else:
        cv2.imshow('Original Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()