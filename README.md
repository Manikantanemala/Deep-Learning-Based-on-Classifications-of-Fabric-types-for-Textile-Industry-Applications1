# Deep-Learning-Based-on-Classifications-of-Fabric-types-for-Textile-Industry-Applications1

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-yellow)

---

## Project Overview
This project focuses on developing a **Convolutional Neural Network (CNN)** model to classify different **fabric defect types** in the textile industry. The goal is to help automate **defect detection** during fabric production using **deep learning** and **image analysis**. The model is trained on labeled fabric images representing multiple defect types and achieves strong classification accuracy.

---

## Dataset Description
The dataset consists of fabric images grouped into multiple defect categories such as:

| Label | Class Name |
|--------|-------------|
| 0 | Corrugation of Rails |
| 1 | Hogging of Rails |
| 2 | No Defects |
| 3 | Rail End Batter |
| 4 | Rail Wear |
| 5 | Scabbing of Rails |
| 6 | Shelling and Black Spots |
| 7 | Wheel Burns |

Each folder in the dataset contains training, validation, and testing images for each category.


---

## ⚙️ Model Architecture
The CNN model is built using **TensorFlow** and **Keras**:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
