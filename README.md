# 🌿 Mulberry Leaf Disease Detection using Deep Learning (CNN)

This project is a deep learning-based system designed to detect diseases in mulberry leaves using image classification. By leveraging Convolutional Neural Networks (CNN), the system can classify leaf images into categories such as **Healthy**, **Leaf Rust**, and **Leaf Spot**.

---

## 📌 Project Highlights

- 🔍 **Automatic Disease Detection** using CNNs
- 📊 Achieved **97% Accuracy**
- 📷 Works with user-uploaded images
- 🧠 Trained on real-world dataset of mulberry leaf images
- 🌐 Optional web UI using **Streamlit**

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **TensorFlow / Keras**
- **NumPy & Pandas**
- **Matplotlib / Seaborn**
- **Streamlit** *(for web interface)*
- **PIL** (Python Imaging Library)

---

## ⚙️ How it Works

1. **Image Upload**  
   User uploads an image of a mulberry leaf via Streamlit UI or script.

2. **Preprocessing**  
   Images are resized, normalized, and converted to the right format.

3. **Prediction**  
   A trained CNN model predicts the leaf condition: *Healthy*, *Rust*, or *Spot*.

4. **Output**  
   The result is displayed with confidence score and optional recommendation.

---

## 🧠 Model Details

- **Architecture**: Custom CNN (4 Conv layers + Dense layers)
- **Activation Functions**: ReLU, Softmax
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Training Accuracy**: ~97%
- **Validation Accuracy**: ~95%

## Data set 
https://www.kaggle.com/datasets/nahiduzzaman13/mulberry-leaf-dataset

---

## 📁 Project Structure


