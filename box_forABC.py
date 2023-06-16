import cv2
import numpy as np

def crop_image(img, x_offset, y_offset):
    half_width, half_height = 60, 60
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
        if w > 0.9 * h and x > img.shape[1] * 0.1 and y > img.shape[0] * 0.1 and x + w < img.shape[1] * 0.9 and y + h < \
                img.shape[0] * 0.9:
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
        # cv2.rectangle(img, (x_min, y_min), (x_min + width, y_min + height), (0, 0, 255), 2)
        x_max = x_min + width
        y_max = y_min + height
        cv2.rectangle(img, (max_rect[0], max_rect[1]), (max_rect[0]+max_rect[2], max_rect[1]+max_rect[3]), (0, 0, 255), 2)
        cropped = img[y_min:y_max, x_min:x_max]
        dim = (647, 373)
        resized = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite('resized.jpg', resized)
        img = cv2.imread('resized.jpg')
        offsets = [
            (img.shape[1] // 2 - 93, img.shape[0] // 2 - 84),
            (img.shape[1] // 2 + 93, img.shape[0] // 2 - 84),
            (img.shape[1] // 2 - 93, img.shape[0] // 2 + 82),
            (img.shape[1] // 2 + 93, img.shape[0] // 2 + 82)
        ]
        for i, offset in enumerate(offsets):
            cropped_image = crop_image(img, *offset)
            cv2.imwrite(f'cropped_{i + 1}.jpg', cropped_image)
            cv2.imshow(f'Cropped {i + 1}', cropped_image)
            cv2.imshow('Resized Image', resized)
    else:
        cv2.imshow('Original Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()