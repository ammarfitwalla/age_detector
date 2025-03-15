import numpy as np
import streamlit as st
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from PIL import Image
import gdown
import os

# Import your AgeEstimator class
from age_estimator import AgeEstimator  # Ensure 'your_script' is the filename containing the AgeEstimator class


def download_weights():
    os.mkdir("weights", exist_ok=True)  # Create a directory to store the weights
    model_path = "weights/weights.pt"
    if not os.path.exists(model_path):  # Check if file already exists
        url = "https://drive.google.com/uc?id=1ZVeWifvJ6OTvqk7UGwLLVpVNRb4FRqiI"
        # url = "https://drive.google.com/file/d/1ZVeWifvJ6OTvqk7UGwLLVpVNRb4FRqiI/view?usp=sharing"
        gdown.download(url, model_path, quiet=False, resume=True, proxy=None)
    return model_path

weights_path = download_weights()


# Load model efficiently (caches it so it's loaded only once)
@st.cache(allow_output_mutation=True)
def load_model(weights_path="weights/weights.pt", device="cpu"):
    return AgeEstimator(weights=weights_path, device=device)

# Load the model
model = load_model()

st.title("🎭 Real-Time Age Detection (Above 18 / Below 18)")

# Sidebar options
option = st.sidebar.radio("Choose Input Mode:", ["Upload Image", "Live Webcam"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an Image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Convert uploaded image to a format suitable for prediction
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)

        # Convert PIL image to NumPy array for processing
        image_np = np.array(image)

        # Predict Age
        st.write("⏳ **Processing Image...**")
        processed_image = model.predict_frame(image_np)

        # Display result
        st.image(processed_image, caption="🎯 Processed Image with Age Detection", use_column_width=True)
        st.success("✅ Prediction Completed!")

### **🔹 2️⃣ Live Webcam Mode**
elif option == "Live Webcam":
    st.write("📷 **Turn on your webcam to detect age in real-time**")

    # WebRTC Configuration (For better performance)
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


    # Define Video Processor for Webcam Streaming
    class AgeDetectionProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")  # Convert frame to NumPy array (BGR format)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for PIL
            img_pil = Image.fromarray(img_rgb)  # Convert to PIL Image
            
            # Predict Age using your model
            processed_img = model.predict_frame(np.array(img_pil))

            # Convert back to OpenCV format
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")


    # Start Webcam Streaming
    webrtc_streamer(
        key="age-detection",
        video_processor_factory=AgeDetectionProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},  # Enable video only
    )