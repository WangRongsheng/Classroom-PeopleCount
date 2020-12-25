import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import math
import cv2


np.random.seed(1)


def load_csv(filename):
    file=open(filename,'r')
    dataset=[]
    csv_reader=csv.reader(file)
    for line in csv_reader:
        if not line:
            continue
        dataset.append(line)
    return dataset


filename="./data_full.csv"
dataset=load_csv(filename)
print(dataset[0])
for i in range(len(dataset)):
    for j in range(len(dataset[0])):
        dataset[i][j]=float(dataset[i][j])
np.random.shuffle(dataset)
print(len(dataset[0]))
X_train,Y_train=list(),list()
X_test,Y_test=list(),list()
for i in range(int(len(dataset)*0.7)):
    X_train.append(dataset[i][0:len(dataset[0])-1])
    Y_train.append(dataset[i][-1])
for i in range(int(len(dataset)*0.7),len(dataset)):
    X_test.append(dataset[i][0:len(dataset[0])-1])
    Y_test.append(dataset[i][-1])

#clf = SVC()
#clf.fit(X_train, Y_train)
#print("Accuracy on training set is : {}".format(clf.score(X_train, Y_train)))
#print("Accuracy on test set is : {}".format(clf.score(X_test, Y_test)))
#Y_test_pred = clf.predict(X_test)
# 随机森林
rf=RandomForestClassifier(n_estimators=1000)
rf.fit(X_train, Y_train)
print("RF Accuracy on training set is : {}".format(rf.score(X_train, Y_train)))
print("RF Accuracy on test set is : {}".format(rf.score(X_test, Y_test)))
# 多层感知机
mlp=MLPClassifier(hidden_layer_sizes=(400,100))
mlp.fit(X_train, Y_train)
print("MLP Accuracy on training set is : {}".format(mlp.score(X_train, Y_train)))
print("MLP Accuracy on test set is : {}".format(mlp.score(X_test, Y_test)))
# SVM
svm=SVC()
svm.fit(X_train, Y_train)
print("SVM Accuracy on training set is : {}".format(svm.score(X_train, Y_train)))
print("SVM Accuracy on test set is : {}".format(svm.score(X_test, Y_test)))
# 朴素贝叶斯
nb=GaussianNB()
nb.fit(X_train,Y_train)
print("Naive Bayes Accuracy on training set is : {}".format(nb.score(X_train, Y_train)))
print("Naive Bayes Accuracy on test set is : {}".format(nb.score(X_test, Y_test)))
# 梯度提升
gb=GradientBoostingClassifier()
gb.fit(X_train,Y_train)
print("GradientBoost Accuracy on training set is : {}".format(gb.score(X_train, Y_train)))
print("GradientBoost Accuracy on test set is : {}".format(gb.score(X_test, Y_test)))
# KNN
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
print("KNN with 3 Neighbors Accuracy on training set is : {}".format(knn.score(X_train, Y_train)))
print("KNN with 3 Neighbors Accuracy on test set is : {}".format(knn.score(X_test, Y_test)))
knn1=KNeighborsClassifier(n_neighbors=10)
knn1.fit(X_train,Y_train)
print("KNN with 10 Neighbors Accuracy on training set is : {}".format(knn1.score(X_train, Y_train)))
print("KNN with 10 Neighbors Accuracy on test set is : {}".format(knn1.score(X_test, Y_test)))

# 先把图片手动分割完毕，分割完得到的所有图片放到一个list里面，这个list里面的每个元素进行HOG过程。test set由新图片生成。
cell_size = 10  # cell的大小
bin_size = 9  # bin的大小
angle_unit = 360 / bin_size  # 360度均分为bin_size个部分


def cell_gradient(cell_magnitude, cell_angle):
    # cell_magnitude代表每个图片所有点的梯度值，cell_magnitude代表每个图片所有点的梯度角度。两者都是二维数组。
    # 求每个cell的梯度
    orientation_centers = [0] * bin_size  # 初始化bin_size维向量
    for k in range(cell_magnitude.shape[0]):  # 遍历图像的长
        for l in range(cell_magnitude.shape[1]):  # 遍历图像的宽
            gradient_strength = cell_magnitude[k][l]  # 提取梯度值
            gradient_angle = cell_angle[k][l]  # 提取梯度角度
            min_angle = int(gradient_angle / angle_unit)%bin_size  # 寻找角度所在区间的下界序号
            max_angle = (min_angle + 1) % bin_size  # 寻找角度所在区间的上界序号
            mod = gradient_angle % angle_unit
            orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))  # 未超出上界的部分加到角度较大的区间中
            orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit))  # 超出的部分加到角度较小的区间
    return orientation_centers  # 返回


def load_images(dirname, a):  # 依次读入测试集目标区域的图片
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


def HOG_extract(img):  # HOG特征提取函数
    img = np.sqrt(img / float(np.max(img)))
    height, width = img.shape  # 图片的形状
    gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # x方向的梯度
    gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # y方向的梯度
    # 计算合梯度和方向
    gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
    gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
    # print(gradient_magnitude.shape, gradient_angle.shape)
    gradient_magnitude = abs(gradient_magnitude)
    cell_gradient_vector = np.zeros((round(height / cell_size), round(width / cell_size), bin_size))
    # print(cell_gradient_vector.shape)
    for i in range(cell_gradient_vector.shape[0]):
        for j in range(cell_gradient_vector.shape[1]):
            cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size,
                             j * cell_size:(j + 1) * cell_size]
            cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
            # print(cell_angle.max())
            cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle)  # 为每个细胞单元构建梯度方向直方图
    # 构建block，并将每个block的特征向量归一化。
    hog_vector = []
    for i in range(cell_gradient_vector.shape[0] - 1):
        for j in range(cell_gradient_vector.shape[1] - 1):
        # 步长为1，从左上角向右向下移动遍历每个block
            block_vector = []
            block_vector.extend(cell_gradient_vector[i][j])
            block_vector.extend(cell_gradient_vector[i][j + 1])
            block_vector.extend(cell_gradient_vector[i + 1][j])
            block_vector.extend(cell_gradient_vector[i + 1][j + 1])
            mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
            magnitude = mag(block_vector)
            if magnitude != 0:
                normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]  # 归一化
                block_vector = normalize(block_vector, magnitude)
            hog_vector.append(block_vector)
    features = sum(hog_vector, [])  # 把每个block的特征向量展平
    features.append(1)
    return features  # 返回864维向量


def on_EVENT_LBUTTONDOWN(event, x, y,flags, param):  # 鼠标点击标定目标区域函数
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        cv2.circle(pic_colorful, (x, y), 1, (255, 0, 0), thickness=-1)  # 点过的地方画一个蓝色圈
        cv2.imshow("image", pic_colorful)  # 标定测试集时展示彩色的图片，而不展示中间的灰度化等过程


a = []
b = []
X_test=[]
pic=cv2.imread('./Photos/test_1.jpg', cv2.IMREAD_GRAYSCALE)  # 读取灰度化图像
# cv2.imshow("image", pic) 展示灰度化图像
# cv2.waitKey(0)
# pic = np.sqrt(pic / float(np.max(pic)))
# cv2.imshow("image", pic) 展示gamma校正后的图像
# cv2.waitKey(0)
pic_colorful=cv2.imread('./Photos/test_1.jpg')
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", pic_colorful)
cv2.waitKey(0)  # 无限等待点击，只有按下Esc键会退出点坐标选取
print(a,b)
images=list()
if len(a)%2!=0:
    a.pop(-1)
    b.pop(-1)
for i in range(0,round(len(a)/2)):
    images.append(pic[b[2 * i]:b[2 * i + 1], a[2 * i]:a[2 * i + 1]])
    # 显示边框
    pic_colorful[b[2 * i] - 1:b[2 * i], a[2 * i]:a[2 * i + 1]] = (0, 255, 0)
    pic_colorful[b[2 * i + 1]:b[2 * i + 1] + 1, a[2 * i]:a[2 * i + 1]] = (0, 255, 0)
    pic_colorful[b[2 * i]:b[2 * i + 1], a[2 * i] - 1:a[2 * i]] = (0, 255, 0)
    pic_colorful[b[2 * i]:b[2 * i + 1], a[2 * i + 1]:a[2 * i + 1] + 1] = (0, 255, 0)
images_resized = [0] * len(images)
for i in range(0,len(images)):
    images_resized[i]=cv2.resize(images[i],(50,70),interpolation=cv2.INTER_AREA)  # 基于局部像素的重采样
    feature=HOG_extract(images_resized[i])
    feature.pop(-1)
    X_test.append(feature)
Y_predict=svm.predict(X_test)
print('Y_predict:',Y_predict)
for i in range(0,len(images)):
    if Y_predict[i]==1:  # 有人时，把目标区域标红
        pic_colorful[b[2 * i] - 1:b[2 * i], a[2 * i]:a[2 * i + 1]] = (0, 0, 255)
        pic_colorful[b[2 * i + 1]:b[2 * i + 1] + 1, a[2 * i]:a[2 * i + 1]] = (0, 0, 255)
        pic_colorful[b[2 * i]:b[2 * i + 1], a[2 * i] - 1:a[2 * i]] = (0, 0, 255)
        pic_colorful[b[2 * i]:b[2 * i + 1], a[2 * i + 1]:a[2 * i + 1] + 1] = (0, 0, 255)
    else:  # 无人时，把目标区域标黄
        pic_colorful[b[2 * i] - 1:b[2 * i], a[2 * i]:a[2 * i + 1]] = (0, 255, 255)
        pic_colorful[b[2 * i + 1]:b[2 * i + 1] + 1, a[2 * i]:a[2 * i + 1]] = (0, 255, 255)
        pic_colorful[b[2 * i]:b[2 * i + 1], a[2 * i] - 1:a[2 * i]] = (0, 255, 255)
        pic_colorful[b[2 * i]:b[2 * i + 1], a[2 * i + 1]:a[2 * i + 1] + 1] = (0, 255, 255)

cv2.imshow("image", pic_colorful)
cv2.waitKey(0)