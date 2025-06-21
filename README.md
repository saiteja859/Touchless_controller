# 🖐️ Touchless AI Volume Controller

## 📅 Project Duration:
**September 2024**

## 📌 Overview
The **Touchless AI Volume Controller** is an AI-powered system designed to control the volume of your device using simple hand gestures — eliminating the need for physical interaction. Built using **Python**, the project leverages cutting-edge **Computer Vision** and **Deep Learning** techniques to deliver a smooth and intuitive user experience.

## 🎯 Features
- ✅ **Real-Time Volume Control** using hand gestures  
- 🧠 Utilizes **Mediapipe** for accurate hand landmark detection  
- 👁️ Powered by **OpenCV** for camera and image processing  
- 🔄 Dynamic volume adjustment based on gesture recognition  
- 💡 Performs well in different lighting environments  
- 🖥️ Clean and interactive UI for real-time feedback  
- 🤖 AI-enhanced interaction that feels futuristic and seamless  

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:**  
  - OpenCV  
  - Mediapipe  
  - Pycaw (for Windows audio control)  
  - NumPy  
  - Math  

## 🚀 How It Works
1. **Camera Feed Activation**: The webcam captures a live video feed.
2. **Hand Detection**: Mediapipe detects and tracks the position of your hand and fingers.
3. **Gesture Interpretation**: The distance between specific fingers (e.g., thumb and index finger) is calculated.
4. **Volume Mapping**: This distance is mapped to a system volume range using Pycaw.
5. **Real-Time Feedback**: Volume changes are shown visually on the screen.



## 💻 Installation

1. **Clone the repository**
  
   git clone https://github.com/yourusername/touchless-volume-controller.git
   cd touchless-volume-controller
   
2. **Install dependencies** -pip install opencv-python mediapipe numpy pycaw

3. **Run the application** -python volume_controller.py
🧠 Use Cases
Hands-free control while cooking, working out, or gaming

Accessibility tool for users with mobility impairments

Smart home or futuristic desktop environments



📌 Future Improvements

Support for custom gestures

Integration with other media applications

Cross-platform audio control support

🙌 Acknowledgements
Mediapipe
OpenCV
Pycaw
