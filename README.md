# Paper Digit

A computer vision algorithm used to detect single numbers, combined with a **custom-built Neural Network** trained on the **MNIST dataset** to identify and predict digits shown through a webcam feed.  

The project integrates **OpenCV** for real-time detection and preprocessing with a **NumPy-based neural network** that can be expanded with user-labeled data.

---

## Features
- **Real-time digit detection** using OpenCV  
- **Neural Network trained on MNIST** for number recognition  
- **Human feedback loop** — capture, label, and grow your dataset interactively  
- **Retraining support** — retrain your model as new samples are added  
- **Preprocessing pipeline** — automatic cropping, resizing, and normalization  

---

## Project Structure
```
paper-digit/
├── captures/                 # Temporary captures (ignored in Git)
├── custom_digits/            # Folder for user-labeled samples (0–9)
├── debug/                    # Debug or visual test images
└── scr/model/
    ├── config.py             # Configurations and parameters
    ├── ex.py                 # Experimental/testing code
    ├── train.py              # Neural network training logic
    ├── webcam.py             # Webcam-based detection and labeling
    ├── weights.pkl           # Saved model weights (ignored in Git)
    ├── train.csv / test.csv  # Data files for training and testing
```

---

## Tech Stack
- **Python 3.10+**
- **OpenCV** — for video capture and preprocessing  
- **NumPy** — for matrix operations and neural network logic  
- **Pandas** — for CSV dataset management  
- **Matplotlib (optional)** — for data visualization  

---

## How to Run

### Install dependencies
```bash
pip install opencv-python numpy pandas matplotlib
```

### Run the webcam digit detector
```bash
python scr/model/webcam.py
```

- Press **SPACE** to capture a frame.  
- The system will ask you to input the digit (0–9).  
- The preprocessed ROI (Region of Interest) is automatically saved to:  
  ```
  custom_digits/<digit>/
  ```

### Retrain the model
After collecting 100–200 labeled samples:
```bash
python scr/model/train.py
```
This retrains the network using both **MNIST** and your **custom digits**.

---

## Model Overview
- Input: 28×28 grayscale images  
- Hidden layers: Configurable in `config.py`  
- Activation: Sigmoid / ReLU  
- Output: 10 neurons (digits 0–9)  
- Optimizer: Gradient Descent  
- Training dataset: MNIST  

---

## Workflow
1. The webcam captures an image.  
2. OpenCV finds the region of interest (ROI) containing the digit.  
3. The ROI is preprocessed (grayscale, resized, normalized).  
4. The neural network predicts the number shown.  
5. You can manually label new digits for future retraining.

---

## Future Enhancements
- Add a **“none” class** to ignore non-digit frames  
- Implement **continuous learning / retraining loop**  
- Integrate **CNN layers** for improved accuracy  
- Add **confidence visualization** or a simple GUI interface  

---

## License
MIT License © 2025 **Atishay Jain**

---

## Author
**Atishay Jain**  
University of Texas at Austin — Mechanical Engineering + Programming and Computation Certificate + Computational Science and Engineering Certficate
