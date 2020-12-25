import cv2
number=1
# 这段代码的效果：同GenerateTrainset_empty

def on_EVENT_LBUTTONDOWN(event, x, y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        cv2.circle(pic, (x, y), 1, (255, 0, 0), thickness=-1)
        #cv2.putText(pic, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 255, 0), thickness=1)
        cv2.imshow("image", pic)


pic = cv2.imread('./FullPhoto/full'+str(number)+'.jpg')

a = []
b = []
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", pic)
cv2.waitKey(0)  # 无限等待点击，只有按下Esc键会退出点坐标选取
print(a,b)
images=list()
if len(a)%2!=0:
    a.pop(-1)
    b.pop(-1)
for i in range(0,round(len(a)/2)):
    images.append(pic[b[2 * i]:b[2 * i + 1], a[2 * i]:a[2 * i + 1]])
    # 显示边框
    pic[b[2 * i] - 1:b[2 * i], a[2 * i]:a[2 * i + 1]] = (0, 255, 0)
    pic[b[2 * i + 1]:b[2 * i + 1] + 1, a[2 * i]:a[2 * i + 1]] = (0, 255, 0)
    pic[b[2 * i]:b[2 * i + 1], a[2 * i] - 1:a[2 * i]] = (0, 255, 0)
    pic[b[2 * i]:b[2 * i + 1], a[2 * i + 1]:a[2 * i + 1] + 1] = (0, 255, 0)
for i in range(0,len(images)):
    images[i]=cv2.resize(images[i],(50,70),interpolation=cv2.INTER_AREA)  # interpolation=cv2.INTER_CUBIC
    cv2.imwrite("./Train_full/"+str(number)+"-"+str(i)+".png",images[i])
cv2.imshow("image", pic)
cv2.waitKey(0)

