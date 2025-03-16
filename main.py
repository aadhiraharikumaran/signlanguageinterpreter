import streamlit as st
import cv2
import time
import base64
import numpy as np
from groq import Groq
from PIL import Image
import tempfile
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq API client
client = Groq(api_key=GROQ_API_KEY)

# Function to encode image to base64
def encode_image(image_path):
    """Encodes an image to base64 format."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Streamlit UI
st.title("Sign Language Detector using Groq Vision")

# Start video capture button
start_button = st.button("Start Video Capture")

if start_button:
    cap = cv2.VideoCapture(0)  # Open webcam
    frames = []
    start_time = time.time()
    duration = 5  # Capture for 5 seconds

    st.write("Recording for 5 seconds... Perform a sign language gesture.")

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video frame.")
            break

        # Convert frame to RGB (for Streamlit display)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show live video
        st.image(frame_rgb, caption="Live Video Feed", use_container_width=True)

        # Store frames
        frames.append(frame)

    cap.release()
    st.success("Video capture complete!")

    # Convert the last frame to an image
    if frames:
        last_frame = frames[-1]  # Use the last frame as the captured image

        # Save frame as image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            Image.fromarray(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)).save(temp_file.name)
            temp_file_path = temp_file.name

        # Display the captured frame
        st.image(last_frame, caption="Captured Frame for Processing", use_container_width=True)

        # Encode image to base64
        base64_image = encode_image(temp_file_path)

        # Send image to Groq API
        with st.spinner("Analyzing video frame..."):
            response = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Think as a sign language detector and identify the sign in the image."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                temperature=0.7,
                max_completion_tokens=512
            )

        # Display result
        result = response.choices[0].message.content
        st.subheader("Detected Sign:")
        st.write(result)
