
📌 README.md — Image Processing Using Convolution, ReLU & Max Pooling (TensorFlow)
🖼️ Image Processing with a Simple CNN Pipeline
This project demonstrates the fundamental operations of a Convolutional Neural Network (CNN) using TensorFlow, applied to a grayscale image. It manually implements the three major CNN components:
✔️ Convolution
✔️ ReLU Activation
✔️ Max Pooling
The goal is to understand how CNNs extract features from images at a basic level.
🎯 Objectives
Load and preprocess a grayscale image
Apply a 3×3 edge detection convolution kernel
Use ReLU activation to highlight positive features
Apply max pooling to reduce dimensionality
Visualize each stage of the pipeline
🧠 Concepts Used
1️⃣ Convolution
Convolution extracts features from images by sliding a kernel across the pixels.
You used a high-pass edge detection filter:
Copy code

[-1 -1 -1
 -1  8 -1
 -1 -1 -1]
This emphasizes areas with strong intensity changes (edges).
2️⃣ ReLU Activation
Applies:
Copy code

f(x) = max(0, x)
It removes negative values, retaining only strong detected features.
3️⃣ Max Pooling
Reduces spatial size while keeping important patterns.
Used:
Window: 2×2
Stride: 2
Type: MAX
🛠️ Tech Stack
Python
TensorFlow
NumPy
Matplotlib
Jupyter Notebook
📂 Project Structure
Copy code

📁 CNN-Image-Processing
│── paju.jpg                 # Input grayscale image
│── cnn_processing.ipynb     # Jupyter Notebook implementation
│── README.md                # Project documentation
🚀 How to Run
Install required libraries:
Copy code
Bash
pip install tensorflow numpy matplotlib
Place your image (paju.jpg) in the project folder.
Run the notebook:
Copy code
Bash
jupyter notebook cnn_processing.ipynb
View outputs for:
Original image
Convolution output
ReLU activation
Max pooled image
📸 Output Overview
Stage
Description
Original Image
Grayscale image after resizing and normalization
Convolution Output
Edges highlighted using the kernel
ReLU Output
Negative values removed, stronger edge visibility
Max Pooling Output
Reduced feature map with preserved key edges
📈 Results
Successfully detected edges using convolution
ReLU highlighted relevant features
Max pooling reduced dimensions while keeping important structures
Demonstrates core mechanics of how CNNs extract features
🔮 Future Improvements
Apply Sobel or Prewitt operators
Extend pipeline to RGB images
Build a full CNN classification model
Experiment with average/global pooling
Add Gaussian smoothing before convolution
📚 References
TensorFlow Docs (tensorflow.org)
NumPy Docs (numpy.org)
Matplotlib Docs (matplotlib.org)
Gonzalez & Woods — Digital Image Processing
⭐ Author
spoorti shinge
