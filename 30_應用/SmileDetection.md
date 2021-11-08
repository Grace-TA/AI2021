# 微笑偵測 (Smile Detection)

## Source

https://github.com/hromi/SMILEsmileD

## Show Directory with Tree Structure

<div align="center">
     <img 
      src="https://user-images.githubusercontent.com/89304181/140705934-4486de2f-3825-4d48-b170-e521d63701e2.png" 
      width="20%" height="20%">
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




