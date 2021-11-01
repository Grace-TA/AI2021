# img.fill Study: 在圖片上畫一條紫紅色的對角線，寬度為 5 px (0,0在最左上角)

## Program

````python

#Lab:CV1
import time
for i in range(0,255,32):
  print('img.fill:',i)
  img.fill(i)
  # 在圖片上畫一條紫紅色的對角線，寬度為 5 px (0,0在最左上角)
  # cv2.line(影像, 開始座標, 結束座標, 顏色, 線條寬度)
  cv2.line(img, (0, 0), (255, 255), (255, 0, 255), 5)
  # 顯示圖片
  cv2_imshow(img)
  time.sleep(2)

````

## Result

![image](https://user-images.githubusercontent.com/89304181/139642215-7cce43fc-8809-4347-9f03-523058ae4468.png)

![image](https://user-images.githubusercontent.com/89304181/139642243-ace6176f-5526-45d4-8cf3-447fcdb1425e.png)

![image](https://user-images.githubusercontent.com/89304181/139642263-f2755ff3-7ce1-45c4-9f33-7a66441f0965.png)
