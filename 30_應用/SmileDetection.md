# 微笑偵測 (Smile Detection)

## Source

https://github.com/hromi/SMILEsmileD

## Show Directory with Tree Structure

<div align="center">
     <img 
      src="https://user-images.githubusercontent.com/89304181/140705934-4486de2f-3825-4d48-b170-e521d63701e2.png" 
      width="30%" height="30%">
    </div>
   

## 建模: CNN with 3 Times Convolution

````python
filters = 32
conv_size = 3 #卷積 3*3
pool_size = 2 #池化 2*2

model = Sequential()
model.add(layers.Conv2D(filters,(conv_size,conv_size),activation="relu",input_shape=trainX.shape[1:]))
model.add(layers.MaxPooling2D((pool_size,pool_size)))
model.add(layers.Conv2D(filters*2,(conv_size,conv_size),activation="relu"))
model.add(layers.MaxPooling2D((pool_size,pool_size)))
model.add(layers.Conv2D(filters,(conv_size,conv_size),activation="relu"))
model.add(layers.MaxPooling2D((pool_size,pool_size)))
model.add(layers.Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(number_of_categories, activation='softmax'))

model.summary()
````

## 繪製訓練歷程圖表

![image](https://user-images.githubusercontent.com/89304181/140706186-d2f111d1-fd6f-4fd1-99f6-c6cb59f761f0.png)

## Experiment 1

![image](https://user-images.githubusercontent.com/89304181/140706346-2294b8a9-08ff-4fef-bbe7-1f6027bff6c2.png)

## Experiment 2

![image](https://user-images.githubusercontent.com/89304181/140706386-e9a4a377-3319-4fcd-8465-e956e61b8df8.png)

## Experiment 3

![image](https://user-images.githubusercontent.com/89304181/140706479-2e90f499-87b2-4dc2-aa50-802c588ed6bb.png)

## Complete Python Code

````python
#110 Obtain your respository by git clone
!git clone https://github.com/hromi/SMILEsmileD.git

#111 List files and folders
!ls -lrt

#112 Install linux package: tree
!apt-get install tree

#113 Show directory with tree structure
!tree SMILEsmileD -L 3

# Commented out IPython magic to ensure Python compatibility.
# #114 Define the path for positive images and negative images
 %%time
# 沒有微笑(負向)圖片數量
from imutils import paths #路徑檔案管理
neg_images = sorted(list(paths.list_images('SMILEsmileD/SMILEs/negatives/negatives7'))) 
print('neg_images QTY:', len(neg_images))
# 微笑(正向)圖片數量
pos_images = sorted(list(paths.list_images('SMILEsmileD/SMILEs/positives/positives7'))) 
print('pos_images QTY:',len(pos_images))

#115 How many positive and negative images what we have?
neg_images = neg_images[:3690]
print(len(neg_images))
print('neg_images QTY:', len(neg_images))
print(len(pos_images))
print('pos_images QTY:',len(pos_images))

#116 製作數據集 [圖片路徑,label] 微笑:1 沒微笑:0
dataset = [(path, 0) for path in neg_images] + [(path, 1) for path in pos_images]
print(dataset[-2])
len(dataset)

#117 Show a simple image
from IPython.display import Image
Image(dataset[-2][0],width=200)

#118 Show a simple image
from IPython.display import Image
Image(dataset[1][0],width=200)

# Commented out IPython magic to ensure Python compatibility.
#120 import the required python package
# %tensorflow_version 1.x
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.io import imread #skimage python影像處理模組
from skimage.measure import block_reduce

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import models
from keras import layers
from sklearn.model_selection import train_test_split

# Commented out IPython magic to ensure Python compatibility.
# #121 處理圖片: 轉灰階, 統一圖片尺寸, 減少數據量(i.e., block_reduce)
%%time
x_train = []
y_train = []
count=0
for path,label in dataset:
    image = cv2.imread(path) #載入圖片
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #轉灰階
    image = cv2.resize(image,(180,192)) #統一圖片尺寸
    # 3x3局部採樣取平均值 類似池化層 減少數據量 
    image = block_reduce(image, block_size=(3, 3), func=np.mean) 
    x_train.append(image)
    y_train.append(label)
    count=count+1
    if count % 50 == 0:
        print(count,'處理完成..',label)

#122 轉成np矩陣
x_train = np.asarray(x_train)  # 
y_train = np.asarray(y_train)

#123 轉成np矩陣
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)

#124 標準化 
x_train = x_train.astype(np.float32) / 255. #標準化
y_train = y_train.astype(np.int32)
print (x_train.dtype, x_train.min(), x_train.max(), x_train.shape)
print (y_train.dtype, y_train.min(), y_train.max(), y_train.shape)

#125 標籤轉 one hot, 隨機打亂順序
from keras.utils import np_utils
number_of_categories = 2 #分類數
# 標籤轉 one hot
y_train = np_utils.to_categorical(y_train, number_of_categories).astype(np.float32)

#隨機打亂順序
indices = np.arange(len(x_train))
temp_x = x_train
temp_y = y_train
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

#126 顯示訓練數據與標籤 (20張, 4*5)
w=10
h=10
fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 5
for i in range(1, columns*rows +1):
    img = x_train[i]
    fig.add_subplot(rows, columns, i)
    plt.title(y_train[i])
    plt.imshow(img)
plt.show() # 顯示訓練數據與標籤

#127 train_test_split: 分割Training: 80%, Testing: 20% (測試集)
x_train = np.expand_dims(x_train, axis=-1)
# 分割20%成為測試集
(trainX, testX, trainY, testY) = train_test_split(x_train, y_train, test_size=0.20, random_state=1)
print(trainX.shape)

#130 建模:CNN with 3 times Convolution
filters = 32
conv_size = 3 #卷積 3*3
pool_size = 2 #池化 2*2

model = Sequential()
model.add(layers.Conv2D(filters,(conv_size,conv_size),activation="relu",input_shape=trainX.shape[1:]))
model.add(layers.MaxPooling2D((pool_size,pool_size)))
model.add(layers.Conv2D(filters*2,(conv_size,conv_size),activation="relu"))
model.add(layers.MaxPooling2D((pool_size,pool_size)))
model.add(layers.Conv2D(filters,(conv_size,conv_size),activation="relu"))
model.add(layers.MaxPooling2D((pool_size,pool_size)))
model.add(layers.Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(number_of_categories, activation='softmax'))

model.summary()

# Commented out IPython magic to ensure Python compatibility.
# #131 模型編譯與訓練
# %%time
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(trainX, trainY, epochs=10)

#132 Model Saving to the working directory
model.save('smile.h5')

#133 Where are you? (i.e., Show your working directory)
!pwd

# Commented out IPython magic to ensure Python compatibility.
#135 Change directory to your required path
# %cd '/content/drive/MyDrive/AI2021/FaceDetection'

#136 Check h5 has been saved in the above directory or not
!ls -lrt *.h5

#140 模型評估 (20% testing data)
score = model.evaluate(testX, testY)
print('Test score:', score[0])
print('Test accuracy:', score[1])

#141 繪製訓練歷程圖表 accuracy不能只寫acc
history_dict = history.history
print(history_dict.keys())
plt.style.use("ggplot")
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["accuracy"], label="accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

#150 Exp 1 
tt = 'VF05.jpg'
test_images = []
t_image = cv2.imread(tt)
t_image = cv2.cvtColor(t_image, cv2.COLOR_BGR2GRAY)
t_image = cv2.resize(t_image,(60,64))
test_images.append(t_image)

plt.imshow(t_image)
test_images = np.asarray(test_images)
test_images = test_images.astype(np.float32) / 255.

test_images = np.expand_dims(test_images, axis=-1)
p = model.predict(np.array([test_images[0]]))[0]

print(p)
class_names = ["Neutral","Smiling"]
bar_width = 50 #刻度寬度
left_count = int(p[1] * bar_width) #使用Smiling決定 左邊刻度
right_count = bar_width - left_count 
left_side = '-' * left_count #顯示左邊長度
right_side = '-' * right_count #顯示右邊長度
print (class_names[0], left_side + '<|>' + right_side, class_names[1])

#151 Exp 2
tt = 'VF04.jpg'
test_images = []
t_image = cv2.imread(tt)
t_image = cv2.cvtColor(t_image, cv2.COLOR_BGR2GRAY)
t_image = cv2.resize(t_image,(60,64))
test_images.append(t_image)

plt.imshow(t_image)
test_images = np.asarray(test_images)
test_images = test_images.astype(np.float32) / 255.

test_images = np.expand_dims(test_images, axis=-1)
p = model.predict(np.array([test_images[0]]))[0]

print(p)
class_names = ["Neutral","Smiling"]
bar_width = 50 #刻度寬度
left_count = int(p[1] * bar_width) #使用Smiling決定 左邊刻度
right_count = bar_width - left_count 
left_side = '-' * left_count #顯示左邊長度
right_side = '-' * right_count #顯示右邊長度
print (class_names[0], left_side + '<|>' + right_side, class_names[1])

#153 Check uploaded file (i.e., *.png)
!ls *.png

#155 #Exp 3
tt = 'Smile.png'
test_images = []
t_image = cv2.imread(tt)
t_image = cv2.cvtColor(t_image, cv2.COLOR_BGR2GRAY)
t_image = cv2.resize(t_image,(60,64))
test_images.append(t_image)

plt.imshow(t_image)
test_images = np.asarray(test_images)
test_images = test_images.astype(np.float32) / 255.

test_images = np.expand_dims(test_images, axis=-1)
p = model.predict(np.array([test_images[0]]))[0]

print(p)
class_names = ["Neutral","Smiling"]
bar_width = 50 #刻度寬度
left_count = int(p[1] * bar_width) #使用Smiling決定 左邊刻度
right_count = bar_width - left_count 
left_side = '-' * left_count #顯示左邊長度
right_side = '-' * right_count #顯示右邊長度
print (class_names[0], left_side + '<|>' + right_side, class_names[1])

````




