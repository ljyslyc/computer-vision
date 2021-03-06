# 实验三：图像融合

### 实验目标

本次实验的目标是使用多分辨率融合技术无缝地融合两幅图像，图像通过轻微的变形和平滑的接缝将两个图像连接在一起。本次实验帮助学生掌握高斯金字塔、拉普拉斯金字塔以及多分辨率图像还原等技术处理过程。

### 注意事项

* 课程群文件的“实验项目-3”目录中提供了本次实验所需要的两张样例数据图像文件。这两张图片可以用来合成经典的“橘苹果”图像。
* 可以通过使用高斯和拉普拉斯堆栈/金字塔来完成图像的融合，但有可能仍然会在结果图像中存在比较明显的接缝。
* 可参考Burt和Adelson在1983年的论文中描述的方法使用遮罩来优化融合处理过程，以提高接缝的平滑性。该论文可参见“实验项目-3“中的spline83.pdf文件。
* 学生应该基于自己选择的图像来生成各种有趣的融合图像，并尝试不同的融合方式或融合区域。

### 改进优化

* 可以尝试使用图像颜色来增强融合效果。

### 实验要求

* 本次实验为必做实验内容。
* 实验要求至少完成3组图像的混合。
* 完成实验报告，对算法实现和效果进行分析描述。“改进优化”部分为选做内容，可根据自身情况进行实现。这部分内容可帮助你获得更好的分数。

### MULTI-RESOLUTION BLENDING:
----
        operations.multiResBlendOp(im1, im2, mask, levels, sigma)
            im1: ndarray, first image
            im2: ndarray, second image
            mask: ndarray, mask, RGB black and white picture
            levels: how many levels for laplacian
            sigma: sigma for laplacian