## 📸 Fine-Tuned Image Classifier
A fine-tuned AlexNet-based image classification model that enhances visual recognition and provides detailed contextual descriptions using Cohere AI.

### 🚀 Features

- **Fine-Tuned AlexNet Model**: Optimized for CIFAR-100 dataset.
- **Image Classification**: Classifies images into 100 categories.
- **Cohere AI Integration**: Generates detailed descriptions of classified images.
- **Streamlit Interface**: User-friendly UI for image upload and classification.
- **Optimized Performance**: Transfer learning and hyperparameter tuning improve accuracy.
- **Google Drive Model Hosting**: Efficient model loading via gdown.

### 📂 Project Structure
```
Fine-Tuned-Image-Classifier/
│-- app.py                # Main Streamlit application
│-- requirements.txt      # Project dependencies
│-- model/                # Model files (Downloaded dynamically)
│-- README.md             # Documentation
```
### **🛠️ Installation**

Clone the repository
```
git clone https://github.com/abaiml/abpro.git
cd abpro
```
Create a virtual environment & activate it
```
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```
Install dependencies
```
pip install -r requirements.txt
```
Run the application
```
streamlit run app.py
```
Open your browser and go to http://localhost:8501/.

### **🔑 Environment Variables**

Create a .streamlit/secrets.toml file and add:
```
COHERE_API_KEY = "your-cohere-api-key"
```
### **🏗️ Model Details**

- Pretrained Model: AlexNet
- Dataset: CIFAR-100
- Fine-Tuned for 100 Classes: Fruits, vegetables, animals, food items, etc.
- Model Downloaded from Google Drive

### **🎯 How It Works**

1. Upload an image.
2. The image is preprocessed and classified.
3. The model returns top-5 predictions with probabilities.
4. Cohere AI generates a detailed description of the classified object.

### **📜 License**
This project is open-source under the MIT License.

### **🤝 Contributing**
Feel free to fork the repository and submit pull requests.

### **📞 Support**
For any issues or feature requests, create an issue in the repository or contact the developer at ayushbora1001@gmail.com.

🚀 Happy Coding! 🎨

