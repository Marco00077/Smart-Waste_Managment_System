# 🌍 AI Waste Classifier

An AI-powered image classifier that identifies whether waste is **biodegradable** or **non-biodegradable** using deep learning (CNN with transfer learning) and a modern Tkinter GUI.

## ✨ Features

- **Transfer Learning** with MobileNetV2 (pre-trained on ImageNet)
- **Modern Dark UI** with intuitive design
- **Real-time Classification** with confidence scores
- **High Accuracy** even with limited training data
- **Easy to Use** - just select an image and classify!

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

The project includes a sample dataset. Train the model:

```bash
python model/train_simple.py
```

This will:
- Load images from `dataset/biodegradable/` and `dataset/non_biodegradable/`
- Train using transfer learning (MobileNetV2)
- Save the model as `waste_classifier_model.h5`
- Save class names to `class_names.txt`

### 3. Run the Application

```bash
python app.py
```

## 📖 Usage

1. Click **"📁 Select Image"** to choose a waste image
2. Click **"🔍 Classify Waste"** to identify the waste type
3. View the result:
   - 🌱 **BIODEGRADABLE** (green) - Can decompose naturally
   - ♻️ **NON-BIODEGRADABLE** (red) - Requires proper disposal
4. See the confidence percentage
5. Click **"🗑️ Clear"** to classify another image

## 📁 Project Structure

```
waste-classifier/
├── dataset/
│   ├── biodegradable/       # Food waste, paper, organic materials
│   └── non_biodegradable/   # Plastic, metal, glass
├── model/
│   ├── simple_classifier.py # Transfer learning model (MobileNetV2)
│   └── train_simple.py      # Training script
├── app.py                   # Tkinter GUI application
├── requirements.txt         # Python dependencies
├── waste_classifier_model.h5 # Trained model (generated)
└── class_names.txt          # Class labels (generated)
```

## 🎯 Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Transfer Learning**: Frozen base + custom classification head
- **Data Augmentation**: Flip, rotation, zoom, contrast adjustment
- **Input Size**: 224x224 RGB images
- **Output**: 2 classes (biodegradable, non-biodegradable)

## 📊 Adding More Training Data

For better accuracy, add more images to the dataset:

### Option 1: Manual Collection
- Take photos with your phone
- Download from Google Images
- Add to `dataset/biodegradable/` or `dataset/non_biodegradable/`

### Option 2: Kaggle Dataset
Download from [Kaggle Waste Classification Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data)

**Recommended**: 100+ images per class for good results

After adding images, retrain:
```bash
python model/train_simple.py
```

## 🛠️ Requirements

- Python 3.8+
- TensorFlow 2.15+
- Tkinter (included with Python)
- PIL/Pillow
- NumPy

## 💡 Tips

- Use clear, well-lit images for best results
- The model works best with images similar to training data
- More training data = better accuracy
- Transfer learning allows good results even with limited data

## 🐛 Troubleshooting

**Model not found:**
```bash
python model/train_simple.py
```

**Low accuracy:**
- Add more training images (100+ per class recommended)
- Ensure balanced dataset (similar number of images per class)

**Tkinter not available (Linux):**
```bash
sudo apt-get install python3-tk
```

## 📝 License

MIT License - Free to use and modify!
