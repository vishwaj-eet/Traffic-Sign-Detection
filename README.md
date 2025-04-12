# 🚦 Real-Time Traffic Sign Recognition using CNN

A real-time traffic sign recognition system built using Convolutional Neural Networks (CNNs) and OpenCV. This project detects and classifies German traffic signs using a webcam and deep learning. Ideal for early-stage intelligent transportation and driver-assistance systems.

---

## 📌 Features

- 🎯 43-class traffic sign classification (GTSRB dataset)
- 🧠 Custom CNN model built from scratch using TensorFlow/Keras
- 📈 Real-time prediction from webcam feed using OpenCV
- 🖼️ Grayscale, histogram equalization & normalization preprocessing
- 🔄 Data augmentation to simulate real-world variability
- 💾 Save/load model with Pickle
- 📊 Accuracy/loss plots for performance evaluation

---

## 🗂️ Project Structure

- Traffic_sign_main.py # Model training script 
- Traffic_sign_test.py # Real-time sign recognition via webcam 
- model_trained.p # Trained CNN model (will be created after running "Traffic_sign_main.py" file)
- labels.csv # Class labels (0–42) 
- myData/ # Image dataset (sorted into folders by class) 
- requirements.txt # Required Python packages 
- README.md # Project documentation

---


## 📊 Dataset

- **Name**: GTSRB (German Traffic Sign Recognition Benchmark)
- **Total Images**: ~43,000
- **Classes**: 43
- **Source**: [GTSRB Website](https://benchmark.ini.rub.de/)

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/vishwaj-eet/Traffic-Sign-Detection.git
cd traffic-sign-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python TrafficSign_Main.py
```

### 4. Test with webcam
```bash
python TrafficSign_Test.py
```
- ⚠️ Make sure your webcam is connected. Ensure model_trained.p exists from the training step.

---

## 🧠 Model Architecture (Layer-wise)

- 🧩 Conv2D (60 filters, 5×5) → ReLU  
- 🧩 Conv2D (60 filters, 5×5) → ReLU  
- 📉 MaxPooling2D (2×2)  
- 🧩 Conv2D (30 filters, 3×3) → ReLU  
- 🧩 Conv2D (30 filters, 3×3) → ReLU  
- 📉 MaxPooling2D (2×2)  
- 🧱 Flatten  
- 🎯 Dense(500) → ReLU  
- 🔁 Dropout(0.5)  
- 🎯 Dense(43) → Softmax

---

## 📦 Requirements

- Python 3.7+
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- scikit-learn
- Matplotlib

---

### 🔧 Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 📈 Sample Output

Real-time recognition via webcam showing predicted class and confidence score:
- CLASS: 1 - STOP 
- PROBABILITY: 99.96%
![image](https://github.com/user-attachments/assets/4b93db89-c657-4c9b-9d51-a21bec4aff1e)

---

## 📃Refernces

- Qiao, X. (2023). *Research on Traffic Sign Recognition Based on CNN Deep Learning Network*. Procedia Computer Science, 228, 826–837. [https://doi.org/10.1016/j.procs.2023.11.102](https://doi.org/10.1016/j.procs.2023.11.102)

- Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012). *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition*. Neural Networks, 32, 323–332. [https://doi.org/10.1016/j.neunet.2012.02.016](https://doi.org/10.1016/j.neunet.2012.02.016)

- Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. arXiv preprint arXiv:1412.6980.

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to document recognition*. Proceedings of the IEEE, 86(11), 2278–2324. [https://doi.org/10.1109/5.726791](https://doi.org/10.1109/5.726791)

- Ciresan, D., Meier, U., & Schmidhuber, J. (2012). *Multi-column deep neural networks for image classification*. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 3642–3649.

- German Traffic Sign Recognition Benchmark (GTSRB). [https://benchmark.ini.rub.de/](https://benchmark.ini.rub.de/)

- Chollet, F. (2015). *Keras: Deep Learning for Humans*. [https://keras.io](https://keras.io)

- Abadi, M., et al. (2016). *TensorFlow: A system for large-scale machine learning*. In *12th USENIX Symposium on Operating Systems Design and Implementation (OSDI)*, 265–283.

- Shorten, C., & Khoshgoftaar, T. M. (2019). *A survey on Image Data Augmentation for Deep Learning*. Journal of Big Data, 6(1), 60.

- Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012). *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition*. Neural Networks, 32, 323–332.

- Cireşan, D. C., Meier, U., Masci, J., & Schmidhuber, J. (2012). *Multi-column deep neural network for traffic sign classification*. Neural Networks, 32, 333–338.

- Sermanet, P., & LeCun, Y. (2011). *Traffic sign recognition with multi-scale convolutional networks*. In *IJCNN*, 2809–2813.

- Zhu, Z., et al. (2016). *Traffic-sign detection and classification in the wild*. In *CVPR*, 2110–2118.

- Houben, S., et al. (2013). *Detection of traffic signs in real-world images: The German Traffic Sign Detection Benchmark*. In *IJCNN*, 1–8.

- Timofte, R., et al. (2009). *Multi-view traffic sign detection, recognition, and 3D localisation*. Machine Vision and Applications, 25(3), 633–647.

- Zhang, L., et al. (2018). *Road sign detection and recognition using multi-scale CNNs*. In *ICPR*, 319–324.

- Yang, G., & Wu, J. (2018). *Traffic sign detection based on Faster R-CNN*. In *ICARCV*, 1857–1861.

- Zhao, Z.-Q., et al. (2019). *Object detection with deep learning: A review*. IEEE TNNLS, 30(11), 3212–3232.

- Redmon, J., & Farhadi, A. (2018). *YOLOv3: An incremental improvement*. arXiv:1804.02767.

- Ren, S., et al. (2015). *Faster R-CNN: Towards real-time object detection*. In *NeurIPS*, 91–99.

- He, K., et al. (2016). *Deep residual learning for image recognition*. In *CVPR*, 770–778.

- Simonyan, K., & Zisserman, A. (2015). *Very deep CNNs for large-scale image recognition*. In *ICLR*.

- Krizhevsky, A., et al. (2012). *ImageNet classification with deep CNNs*. In *NeurIPS*, 1097–1105.

- Szegedy, C., et al. (2015). *Going deeper with convolutions*. In *CVPR*, 1–9.

- LeCun, Y., et al. (1998). *Gradient-based learning applied to document recognition*. Proc. IEEE, 86(11), 2278–2324.

- Kingma, D. P., & Ba, J. (2014). *Adam: A method for stochastic optimization*. arXiv:1412.6980.

- Abadi, M., et al. (2016). *TensorFlow: A system for large-scale machine learning*. In *OSDI*, 265–283.

- Chollet, F. (2015). *Keras: Deep learning library for humans*. [https://keras.io](https://keras.io)

- Shorten, C., & Khoshgoftaar, T. M. (2019). *A survey on image data augmentation for deep learning*. Journal of Big Data, 6(1), 60.

- Ciresan, D., et al. (2011). *Flexible, high performance convolutional neural networks for image classification*. In *IJCAI*, 1237–1242.

- Ramirez, P., et al. (2021). *Automatic traffic sign recognition using deep learning for autonomous vehicles*. Sensors, 21(7), 2234.

- Mohamed, N., et al. (2021). *Real-time traffic sign recognition using YOLOv3*. Procedia Computer Science, 184, 104–111.

- Rani, R., & Duhan, M. (2021). *Traffic sign detection and recognition using YOLOv3 and Faster R-CNN*. Int. J. Comput. Appl., 43(8), 45–53.

- German Traffic Sign Recognition Benchmark (GTSRB). [https://benchmark.ini.rub.de/](https://benchmark.ini.rub.de/)

- Qiao, X. (2023). *Research on Traffic Sign Recognition Based on CNN Deep Learning Network*. Procedia Computer Science, 228, 826–837.

- Albawi, S., Mohammed, T. A., & Al-Zawi, S. (2017). *Understanding of a convolutional neural network*. In *ICET*, 1–6.

- Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). *SegNet: A deep convolutional encoder-decoder architecture for semantic pixel-wise segmentation*. IEEE TPAMI, 39(12), 2481–2495.

- Huang, G., et al. (2017). *Densely connected convolutional networks*. In *CVPR*, 4700–4708.

- Panwar, H., et al. (2020). *Deep learning for traffic sign detection and classification*. IEEE Access, 8, 130537–130546.

- Liu, C., et al. (2022). *A review of traffic sign recognition using deep learning*. Journal of Intelligent Transportation Systems, 26(5), 429–447.

- Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). *YOLOv4: Optimal speed and accuracy of object detection*. arXiv:2004.10934.

- Lu, Z., et al. (2022). *Attention-based networks for traffic sign classification*. Pattern Recognition Letters, 159, 1–8.




