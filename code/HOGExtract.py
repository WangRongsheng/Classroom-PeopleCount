import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def cell_gradient(cell_magnitude, cell_angle):
    orientation_centers = [0] * bin_size
    for k in range(cell_magnitude.shape[0]):
        for l in range(cell_magnitude.shape[1]):
            gradient_strength = cell_magnitude[k][l]
            gradient_angle = cell_angle[k][l]
            min_angle = int(gradient_angle / angle_unit)%8
            max_angle = (min_angle + 1) % bin_size
            mod = gradient_angle % angle_unit
            orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
            orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))
    return orientation_centers


def load_images(dirname, a):
    # a=[36,36,5] for empty
    # a=[34,21,27] for full
    # 因为文件被命名为1-0，1-1，...，3-12
    img_list=[]
    for i in range(len(a)):
        for j in range(a[i]):
            path='./'+dirname+'/'+str(i+1)+'-'+str(j)+'.png'
            img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_list.append(img)
    return img_list

img_list1=load_images('Train_full',[34,21,27])
img_list2=load_images('Train_empty',[36,36,5])
img_list=img_list1+img_list2
dataset=[]
print(len(img_list))
print(len(img_list1))
for ii in range(len(img_list)):
    # ./Train_full/1-0.png
    #cv2.imshow('Image', img)
    #cv2.imwrite("Image-test.jpg", img)
    #cv2.waitKey(0)
    img_list[ii] = np.sqrt(img_list[ii] / float(np.max(img_list[ii])))
    #cv2.imshow('Image', img)
    #cv2.imwrite("Image-test2.jpg", img)
    #cv2.waitKey(0)
    height, width = img_list[ii].shape
    gradient_values_x = cv2.Sobel(img_list[ii], cv2.CV_64F, 1, 0, ksize=5)
    gradient_values_y = cv2.Sobel(img_list[ii], cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
    gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
    #print(gradient_magnitude.shape, gradient_angle.shape)
    cell_size = 10
    bin_size = 9
    angle_unit = 360 / bin_size
    gradient_magnitude = abs(gradient_magnitude)
    cell_gradient_vector = np.zeros((round(height / cell_size), round(width / cell_size), bin_size))
    #print(cell_gradient_vector.shape)
    for i in range(cell_gradient_vector.shape[0]):
        for j in range(cell_gradient_vector.shape[1]):
            cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size,
                             j * cell_size:(j + 1) * cell_size]
            cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
            #print(cell_angle.max())
            cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle)
    hog_image= np.zeros([height, width])
    cell_gradients = cell_gradient_vector
    cell_width = cell_size / 2
    max_mag = np.array(cell_gradients).max()
    for x in range(cell_gradients.shape[0]):
        for y in range(cell_gradients.shape[1]):
            cell_grad = cell_gradients[x][y]
            cell_grad /= max_mag
            angle = 0
            angle_gap = angle_unit
            for magnitude in cell_grad:
                angle_radian = math.radians(angle)
                x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
                y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
                x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
                y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
                cv2.line(hog_image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                angle += angle_gap
    #plt.imshow(hog_image, cmap=plt.cm.gray)
    #plt.show()
    hog_vector = []
    for i in range(cell_gradient_vector.shape[0] - 1):
        for j in range(cell_gradient_vector.shape[1] - 1):
            block_vector = []
            block_vector.extend(cell_gradient_vector[i][j])
            block_vector.extend(cell_gradient_vector[i][j + 1])
            block_vector.extend(cell_gradient_vector[i + 1][j])
            block_vector.extend(cell_gradient_vector[i + 1][j + 1])
            mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
            magnitude = mag(block_vector)
            if magnitude != 0:
                normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                block_vector = normalize(block_vector, magnitude)
            hog_vector.append(block_vector)
    #print('kkk:',np.array(hog_vector).shape)
    features=sum(hog_vector,[])
    if ii<len(img_list1):
        features.append(1)
    else:
        features.append(0)
    dataset.append(features)


output=open('./data_full.csv','w',encoding='gbk')
for i in range(len(dataset)):
    for j in range(len(dataset[1])):
        output.write(str(dataset[i][j]))
        if j != len(dataset[1])-1:
            output.write(',')
    if i != len(dataset)-1:
        output.write('\n')
output.close()
