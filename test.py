import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'Lesson 3/hinh-anh-que-huong-ninh-binh-lung-linh-nhat.jpg'
original_image = cv2.imread(image_path)

# Chuyển đổi hình ảnh sang thang độ xám
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Tính toán biểu đồ của hình ảnh thang độ xám
hist, bins = np.histogram(gray_image.flatten(), bins=256, range=[0, 256])

# Tính hàm phân phối tích lũy (CDF)
cdf = hist.cumsum()
cdf_normalized = cdf * (255 / cdf[-1])

# Áp dụng cân bằng biểu đồ
cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
equalized_image = cdf_final[gray_image]

# Tăng cường độ tương phản bằng cách ánh xạ 5% pixel thành màu đen và trắng thuần túy
percentile_min = np.percentile(equalized_image, 5)
percentile_max = np.percentile(equalized_image, 95)
contrast_stretched = np.clip((equalized_image - percentile_min) * 255 / 
                              (percentile_max - percentile_min), 0, 255).astype('uint8')

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")

axes[0, 1].imshow(gray_image, cmap="gray")
axes[0, 1].set_title("Grayscale Image")
axes[0, 1].axis("off")

axes[0, 2].plot(hist)
axes[0, 2].set_title("Histogram")
axes[0, 2].set_xlim([0, 255])

axes[1, 0].imshow(equalized_image, cmap="gray")
axes[1, 0].set_title("Equalized Image")
axes[1, 0].axis("off")

axes[1, 1].imshow(contrast_stretched, cmap="gray")
axes[1, 1].set_title("Contrast Enhanced")
axes[1, 1].axis("off")

hist_eq, _ = np.histogram(equalized_image.flatten(), bins=256, range=[0, 256])
axes[1, 2].plot(hist_eq)
axes[1, 2].set_title("Equalized Histogram")
axes[1, 2].set_xlim([0, 255])

plt.tight_layout()
plt.show()