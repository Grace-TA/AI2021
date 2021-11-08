# 深度學習 + 人臉辨識應用: 戴口罩提醒

> Reference: 百度 paddlehub

## 1. 口罩偵測

### Exp 1: Detected 3 faces

### Before Test

![image](https://user-images.githubusercontent.com/89304181/140671915-87573ba4-ca0b-475d-bc6f-10dd863d39ab.png)

### After Test

![image](https://user-images.githubusercontent.com/89304181/140671928-888bd5f5-3591-4fe8-a1e8-95c992805a59.png)

---

### Exp 2: Detected 21 faces

### Before Test

![image](https://user-images.githubusercontent.com/89304181/140671940-0a6ca1ab-d3e2-4ded-aee7-bfbf9133d75d.png)

### After Test

![image](https://user-images.githubusercontent.com/89304181/140671952-8308fbfa-149a-47f0-a4f2-88394fd83d10.png)

---

### Exp 3: Detected 17 faces

### Before Test

![image](https://user-images.githubusercontent.com/89304181/140671965-6d6443e0-b531-4d1f-80f3-0f61d2902790.png)

### After Test

![image](https://user-images.githubusercontent.com/89304181/140671979-9bad79d1-fd9f-47f6-a781-bf7185dce101.png)

---

## 2. 戴口罩提醒

### 人臉判定與編號 (+ enumerate()) + Face Mask Detection

#### Demo Result

![image](https://user-images.githubusercontent.com/89304181/140679304-9a0cd078-250c-4afe-afb4-dcf91ba38840.png)

#### Code

````python
from google.colab.patches import cv2_imshow
import cv2
image = cv2.imread(pic)
result = results[0]
for ii, item in enumerate(result['data']):
    x1 = item['left']
    y1 = item['top']
    x2 = item['right']
    y2 = item['bottom']
    kz = item['label']
    image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(image, kz+'_'+str(ii), (x1 - 5, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)
GrayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
draw_1 = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
cv2_imshow(draw_1) 

````

#### 使用if自動判斷 是否配戴口罩 + 編號 (+ enumerate())

![image](https://user-images.githubusercontent.com/89304181/140679590-645a4cc2-157c-4e80-879f-9fcffc256f0f.png)


