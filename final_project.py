import streamlit as st
from PIL import Image
import os
import base64
import pytesseract
from gtts import gTTS
import tempfile
from dotenv import load_dotenv
from io import BytesIO
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

llm = GoogleGenerativeAI(model="gemini-1.5-pro", api_key=GEMINI_API_KEY)

# Set up Streamlit page
st.set_page_config(page_title="SightAssist", layout="wide")
st.title("👁️ Visionary Assist - AI Assistant for Visually Impaired")

# Display features under the main title
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### ⚙️ Features
    - **Real-time scene understanding**  
    - **Text-to-speech conversion for reading visual content**  
    """)

with col2:
    st.markdown("""
    ###  
    - **Personalized assistance for daily tasks**  
    """)

# Functions for functionality

def generate_scene_description(input_prompt, image_data):
    """Generates a scene description using Google Generative AI."""
    llm = genai.GenerativeModel("gemini-1.5-pro")
    response = llm.generate_content([input_prompt, image_data[0]])
    return response.text

def extract_text_from_image(image):
    """Extracts text from the given image using OCR."""
    return pytesseract.image_to_string(image)

def text_to_speech(text,filename="output.mp3"):
    """Converts the given text to speech using gTTS."""
    tts = gTTS(text) 
    with tempfile.NamedTemporaryFile(delete=True) as fp: 
        tts.save(f"{fp.name}.mp3") 
        st.audio(f"{fp.name}.mp3", format="audio/mp3")

def provide_personalized_assistance(image_data): 
    """Provides task-specific guidance based on the uploaded image.""" 
    input_prompt = """
    You are an AI assistant helping visually impaired individuals with daily tasks. Provide: 
    1. Identification of items in the image. 
    2. Reading of any visible labels or text. 
    3. Context-specific information or instructions related to the items.
    """ 
    response = generate_scene_description(input_prompt, image_data) 
    return response


def input_image_setup(uploaded_file):
    """Prepares the uploaded image for processing."""
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data,
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded.")
    
# Upload Image Section
st.markdown("<h3 class='feature-header'>📤 Upload an Image</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag and drop or browse an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

if uploaded_file:
    # Convert the uploaded image to base64
    buffered = BytesIO()
    image = Image.open(uploaded_file)
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()

# Buttons Section
col1, col2, col3,col4 = st.columns(4)

scene_button = col1.button("🌍 Real-Time Scene")
ocr_button = col2.button("📄 Extract Text")
tts_button = col3.button("🎙️ Text-to-Speech")
assist_button = col4.button("🤖 Personalized Assistance")

# Input Prompt for Scene Understanding
input_prompt = """
You are an AI assistant helping visually impaired individuals by describing the scene in the image. Provide:
1. List of items detected in the image with their purpose.
2. Overall description of the image.
3. Suggestions for actions or precautions for the visually impaired.
"""

# Process user interactions
if uploaded_file:
    image = Image.open(uploaded_file)
    image_data = input_image_setup(uploaded_file)

    if scene_button:
        with st.spinner("Generating scene description..."):
            response = generate_scene_description(input_prompt, image_data)
            st.markdown("<h3 class='feature-header'>" "🌍 Scene Description</h3>", unsafe_allow_html=True)
            st.write(response)
            #Save scene description as speech 
            text_to_speech(response, "scene_description.mp3") 
            st.success("✅ Scene description saved as speech!")

    if ocr_button:
        with st.spinner("Extracting text from the image..."):
            text = extract_text_from_image(image)
            st.markdown("<h3 class='feature-header'>📝 Extracted Text</h3>", unsafe_allow_html=True)
            st.text_area("Extracted Text", text, height=150)

    if tts_button:
        with st.spinner("Converting text to speech..."):
            text = extract_text_from_image(image)
            if text.strip():
                text_to_speech(text)
                st.success("✅ Text-to-Speech Conversion Completed!")
            else:
                st.warning("No text found to convert.")
    if assist_button: 
        with st.spinner("Providing personalized assistance..."): 
            assistance = provide_personalized_assistance(image_data) 
            st.markdown("<h3 class='feature-header'>🤖 Personalized Assistance</h3>", unsafe_allow_html=True)
            st.write(assistance) 
            # Save personalized assistance as speech 
            text_to_speech(assistance, "personalized_assistance.mp3") 
            st.success("✅ Personalized assistance saved as speech!")

else:
    st.info("Please Upload Image to Proceed...!")  


# Footer
st.markdown(
    """
    <hr>
    <footer style="text-align:center;">
        <p>Powered by <strong>Google Gemini API</strong> </p>
    </footer>
    """,
    unsafe_allow_html=True,
)
