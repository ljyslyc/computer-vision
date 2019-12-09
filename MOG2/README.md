## MOG2定位物体轮廓
---
MOG2算法，也是高斯混合模型分离算法，是MOG的改进算法。它基于Z.Zivkovic发布的两篇论文，即2004年发布的“Improved adaptive Gausian mixture model for background subtraction”和2006年发布的“Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction”中提出。该算法的一个重要特征是 它为每个像素选择适当数量的高斯分布，它可以更好地适应不同场景的照明变化等。
```python
bs = cv2.createBackgroundSubtractorMOG2(history=history, detectShadows=True)
bs.setHistory(history)
fg_mask = bs.apply(frame)
```