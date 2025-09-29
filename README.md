# IMAGE-TRANSFORMATIONS
## NAME : PRAVESH N
## REG NO : 212223230124
## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
## Step1:
Import necessary libraries such as OpenCV, NumPy, and Matplotlib for image processing and visualization.

## Step2:
Read the input image using cv2.imread() and store it in a variable for further processing.

## Step3:
Apply various transformations like translation, scaling, shearing, reflection, rotation, and cropping by defining corresponding functions:

1.Translation moves the image along the x or y-axis. 2.Scaling resizes the image by scaling factors. 3.Shearing distorts the image along one axis. 4.Reflection flips the image horizontally or vertically. 5.Rotation rotates the image by a given angle.

## Step4:
Display the transformed images using Matplotlib for visualization. Convert the BGR image to RGB format to ensure proper color representation.

## Step5:
Save or display the final transformed images for analysis and use plt.show() to display them inline in Jupyter or compatible environments.

## Program:
````
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()

image_path = list(uploaded.keys())[0]
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

rows, cols, _ = image.shape

M_translate = np.float32([[1, 0, 50], [0, 1, 100]])
translated_image = cv2.warpAffine(image_rgb, M_translate, (cols, rows))

scaled_image = cv2.resize(image_rgb, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

M_shear = np.float32([[1, 0.5, 0], [0.5, 1, 0]])
sheared_image = cv2.warpAffine(image_rgb, M_shear, (int(cols * 1.5), int(rows * 1.5)))

reflected_image = cv2.flip(image_rgb, 1)

M_rotate = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
rotated_image = cv2.warpAffine(image_rgb, M_rotate, (cols, rows))

cropped_image = image_rgb[50:300, 100:400]

plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(translated_image)
plt.title("Translated Image")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(scaled_image)
plt.title("Scaled Image")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(sheared_image)
plt.title("Sheared Image")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(reflected_image)
plt.title("Reflected Image")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(rotated_image)
plt.title("Rotated Image")
plt.axis('off')

plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(cropped_image)
plt.title("Cropped Image")
plt.axis('off')
plt.show()



``````
## Output:
## ORIGINAL IMAGE

<img width="390" height="237" alt="Screenshot 2025-09-29 093646" src="https://github.com/user-attachments/assets/eca4fb33-ea33-4256-b523-91844189adbe" />


### i)Image Translation
<br>

<img width="391" height="240" alt="Screenshot 2025-09-29 093729" src="https://github.com/user-attachments/assets/7d55668b-e80e-45da-a5fc-f9f4dd4d0261" />


<br>
<br>
<br>

### ii) Image Scaling
<br>

<img width="380" height="242" alt="Screenshot 2025-09-29 093805" src="https://github.com/user-attachments/assets/50655586-14cc-4d9c-a4dd-aac93323fcfb" />


<br>
<br>
<br>


### iii)Image shearing
<br>

<img width="388" height="234" alt="Screenshot 2025-09-29 093854" src="https://github.com/user-attachments/assets/0b6a25c7-0487-4f9f-88f3-2cea7a657d59" />

<br>
<br>
<br>


### iv)Image Reflection
<br>

<img width="385" height="242" alt="Screenshot 2025-09-29 094018" src="https://github.com/user-attachments/assets/adff2c43-e5a7-4f21-b18f-a07be2b20a5d" />


<br>
<br>
<br>



### v)Image Rotation
<br>

<img width="374" height="239" alt="Screenshot 2025-09-29 094041" src="https://github.com/user-attachments/assets/a720469d-ca66-4b9a-9876-3e0cab8f0db4" />

<br>
<br>
<br>



### vi)Image Cropping
<br>

<img width="409" height="361" alt="Screenshot 2025-09-29 094103" src="https://github.com/user-attachments/assets/132bc40c-5917-4337-91b0-068f400ce559" />

<br>
<br>
<br>


## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
