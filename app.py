import torch
from torchvision import models, transforms
from PIL import Image
import cohere
import streamlit as st
import gdown
import os

# Retrieve the API key from Streamlit secrets
cohere_api_key = st.secrets["COHERE_API_KEY"]

# Initialize Cohere with the API key
cohere_client = cohere.Client(cohere_api_key)

# Google Drive file ID (replace with your actual file ID)
file_id = "1NFfdK_U5yQE1S4q8S4cezvhat9qyIFAx"  # Your actual file ID
model_url = f"https://drive.google.com/uc?id={file_id}"

# Path where model will be saved (persistent across sessions)
model_file_path = "fine_tuned_alexnet_cifar100.pth"

# Function to download the model only if not already present
def download_model():
    if not os.path.exists(model_file_path):
        # Download the model only if it doesn't already exist locally
        gdown.download(model_url, model_file_path, quiet=False)

# Cache the model download to prevent multiple downloads
@st.cache_resource
def load_model():
    # Ensure the model is downloaded once before loading it
    download_model()
    
    # Load the checkpoint
    checkpoint = torch.load(model_file_path, map_location=torch.device('cpu'))
    
    # Initialize the model architecture (AlexNet)
    alexnet = models.alexnet(pretrained=False)
    
    # Modify the classifier to have 100 output classes (for CIFAR-100)
    alexnet.classifier[6] = torch.nn.Linear(in_features=4096, out_features=100)
    
    # Load only the model weights (not the optimizer or epoch states)
    model_state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    alexnet.load_state_dict(model_state_dict)
    
    alexnet.eval()
    return alexnet
        
# Load the model
alexnet = load_model()

# CIFAR-100 class labels (100 classes)
cifar100_labels = [
    'apple', 'orange', 'banana', 'pear', 'grape', 'watermelon', 'peach', 'apricot', 'lemon', 'lime',
    'cherry', 'strawberry', 'blueberry', 'blackberry', 'raspberry', 'grapefruit', 'cantaloupe', 'fig', 
    'pineapple', 'pomegranate', 'plum', 'apricot', 'tomato', 'potato', 'carrot', 'lettuce', 'onion', 
    'broccoli', 'cabbage', 'spinach', 'pumpkin', 'squash', 'pea', 'bean', 'eggplant', 'zucchini', 'peppers', 
    'peas', 'corn', 'greenbean', 'kale', 'brussels', 'radish', 'sweet potato', 'cucumber', 'garlic', 'ginger',
    'basil', 'oregano', 'rosemary', 'thyme', 'pasta', 'tofu', 'burger', 'fries', 'pizza', 'sandwich', 'wrap',
    'ketchup', 'mustard', 'cheese', 'chocolate', 'bacon', 'chicken', 'fish', 'steak', 'beef', 'sausage', 
    'salmon', 'mushroom', 'cornbread', 'donut', 'croissant', 'bagel', 'coffee', 'tea', 'juice', 'water', 
    'milk', 'beer', 'vodka', 'wine', 'champagne', 'soda', 'cocktail', 'whiskey', 'whipcream', 'icecream', 
    'yogurt', 'cereal', 'muffin', 'cookie', 'cake', 'pie', 'popcorn', 'candy', 'chocolatechip', 'gingerbread', 
    'cinnamon', 'brownie', 'applepie', 'blueberrymuffin', 'waffle', 'pancake', 'fruitcake', 'pastry', 'lollipop'
]

# Function to preprocess the image for AlexNet (CIFAR-100 fine-tuned model)
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image

# Generate text based on the classification result using Cohere
def generate_text_cohere(prompt):
    # Call the generate method on the cohere_client instance
    response = cohere_client.generate(
        model="command-xlarge-nightly",  
        prompt=prompt,
        max_tokens=500,  
        temperature=0.75
    )
    return response.generations[0].text.strip()

# Function to classify the image using AlexNet and return top N predictions
def classify_image(image_tensor, top_n=3):
    with torch.no_grad():
        outputs = alexnet(image_tensor)
    
    # Get top N predictions
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    top_probabilities, top_indices = torch.topk(probabilities, top_n)
    
    # Get the class labels for the top N predictions
    top_labels = [cifar100_labels[idx] for idx in top_indices[0]]
    top_probs = top_probabilities[0].tolist()
    
    return top_labels, top_probs

# Main function to process the image, classify it, and generate a description
def process_and_generate_text(image):
    # Preprocess and classify the image
    image_tensor = preprocess_image(image)
    top_labels, top_probs = classify_image(image_tensor, top_n=3)
    
    # Select the highest probability label as the final classification
    label_name = top_labels[0] 
    prompt = f"This image shows a {label_name}. Describe it in detail."
    
    # Generate descriptive text using Cohere
    description = generate_text_cohere(prompt)
    
    return label_name.upper(), description, top_labels, top_probs

# Streamlit application interface
st.title("Image Description Generator using Fine-tuned AlexNet and Cohere")

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Generate and display the description
    with st.spinner("Processing..."):
        label_name, description, top_labels, top_probs = process_and_generate_text(image)
    
    # Display the result with clear formatting
    st.subheader("Generated Description")
    st.write(f"**Detected as** :  {label_name}")
    
    # Show the top 3 predictions and their probabilities 
    st.write("**Top 3 Predicted Classes and Probabilities:**")
    top_predictions = ""
    for i in range(3):
        top_predictions += f"{i+1}.**{top_labels[i]}** : {top_probs[i]*100:.2f}% {'&nbsp;' * 15} "
    
    # Use markdown to display side by side with enough space
    st.markdown(top_predictions, unsafe_allow_html=True)
    
    # Display full description in an expandable area for better readability
    with st.expander("Click to view full description"):
        st.write(description)
