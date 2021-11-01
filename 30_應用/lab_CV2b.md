# cv2.line Study

## Code
````python
img.fill(20)
# 在圖片上畫一條紫紅色的對角線，寬度為 5 px (0,0在最左上角)
# cv2.line(影像, 開始座標, 結束座標, 顏色, 線條寬度)
for i in range(0,255,64):
  print('(i, i), (i+64, i+64) at ',i)
  cv2.line(img, (i, i), (i+10, i+10), (0, 0, 255), 5)
  cv2_imshow(img) # 顯示圖片
````

## Result

![image](https://user-images.githubusercontent.com/89304181/139643527-e4410e48-98a2-4162-a9e9-6efe59bd12d3.png)

![image](https://user-images.githubusercontent.com/89304181/139643572-80a58a4f-ac9d-4022-9963-3f6e56a4e796.png)
