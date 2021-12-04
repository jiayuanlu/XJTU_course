import numpy as np
import cv2

def biFilter(img, r, color, space) : 
    space_weight = []     
    space_weight_row = []
    space_weight_col = []
    R, G, B = cv2.split(img)
    R_channel, G_channel, B_channel = cv2.split(img)
    height = len(R)
    width = len(R[0])
    color_coef = -1 / (2*color **2)
    color_weight = []       
    for i in range(256) :
        color_weight.append(np.exp(i * i * color_coef))
    space_coef = -1 / (2* space **2)
    Max = 0
    for i in range(-r, r+1) :
        for j in range(-r, r+1) :
            r_block = i**2 + j**2
            space_weight.append(np.exp(r_block * space_coef))
            space_weight_row.append(i)
            space_weight_col.append(j)
            Max = Max + 1
    for row in range(height) :
        for col in range(width) :
            value = 0
            weight = 0
            for i in range(Max) :
                hang = row + space_weight_row[i]
                lie = col + space_weight_col[i]
                if hang < 0 or lie < 0 or hang >= height or lie >= width :
                    pixel = 0
                else :
                    pixel = R[hang][lie]
                w = np.float32(space_weight[i]) * np.float32(color_weight[np.abs(pixel - R[row][col])])
                value = value + pixel * w
                weight = weight + w
            R_channel[row][col] = np.uint8(value / weight)
    for row in range(height) :
        for col in range(width) :
            value = 0
            weight = 0
            for i in range(Max) :
                hang = row + space_weight_row[i]
                lie = col + space_weight_col[i]
                if hang < 0 or lie < 0 or hang >= height or lie >= width :
                    pixel = 0
                else :
                    pixel = G[hang][lie]
                w = np.float32(space_weight[i]) * np.float32(color_weight[np.abs(pixel - G[row][col])])
                value = value + pixel * w
                weight = weight + w
            G_channel[row][col] = np.uint8(value / weight)
    for row in range(height) :
        for col in range(width) :
            value = 0
            weight = 0
            for i in range(Max) :
                hang = row + space_weight_row[i]
                lie = col + space_weight_col[i]
                if hang < 0 or lie < 0 or hang >= height or lie >= width :
                    pixel = 0
                else :
                    pixel = B[hang][lie]
                w = np.float32(space_weight[i]) * np.float32(color_weight[np.abs(pixel - B[row][col])])
                value = value + pixel * w
                weight = weight + w
            B_channel[row][col] = np.uint8(value / weight)
    cv2.imshow("8_after", cv2.merge([R_channel, G_channel, B_channel]))
    cv2.imwrite("8_after.png", cv2.merge([R_channel, G_channel, B_channel]))
 
img = cv2.imread("8.jpeg")
cv2.imshow("srcimage", img)
biFilter(img, 5, 120, 160)
cv2.waitKey(0)