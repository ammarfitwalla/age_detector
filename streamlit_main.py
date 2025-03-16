import cv2
import numpy as np
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from PIL import Image
import gdown
import os

# Import your AgeEstimator class
from age_estimator_cv2 import AgeEstimator  # Ensure 'age_estimator.py' contains AgeEstimator class


def download_weights():
    os.makedirs("weights", exist_ok=True)  # Create a directory if it doesn't exist
    model_path = "weights/weights.pt"
    if not os.path.exists(model_path):  # Check if file already exists
        url = "https://drive.google.com/uc?id=1ZVeWifvJ6OTvqk7UGwLLVpVNRb4FRqiI"
        gdown.download(url, model_path, quiet=False, resume=True)
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

### **🔹 1️⃣ Image Upload Mode**
if option == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an Image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Convert uploaded image to a format suitable for prediction
        image = Image.open(uploaded_file).convert("RGB")  # Ensure it's RGB
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)

        # Convert PIL image to NumPy array
        image_np = np.array(image)

        # Predict Age
        st.write("⏳ **Processing Image...**")
        processed_image = model.predict_frame(image_np)

        # Convert NumPy array back to PIL image for display
        processed_image_pil = Image.fromarray(processed_image)

        # Display result
        st.image(processed_image_pil, caption="🎯 Processed Image with Age Detection", use_column_width=True)
        st.success("✅ Prediction Completed!")

### **🔹 2️⃣ Live Webcam Mode**
elif option == "Live Webcam":
    # st.write("📷 **Turn on your webcam to detect age in real-time**")

    # # WebRTC Configuration (For better performance)
    # rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    # # Define Video Processor for Webcam Streaming
    # class AgeDetectionProcessor(VideoProcessorBase):
    #     def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
    #         # Convert frame to PIL image
    #         # img_pil = Image.fromarray(frame.to_ndarray(format="rgb24"))  # Convert to RGB format

    #         # # Predict Age using your model
    #         # processed_img = model.predict_frame(np.array(img_pil))

    #         # # Convert back to PIL and then NumPy
    #         # processed_img_pil = Image.fromarray(processed_img)

    #         # # Convert PIL image back to OpenCV format for WebRTC (bgr24 is required)
    #         # return av.VideoFrame.from_ndarray(np.array(processed_img_pil), format="rgb24")

    #         img = frame.to_ndarray(format="bgr24")  # OpenCV uses BGR
    #         # Process the image using the model
    #         processed_img = model.predict_frame(img)

    #         # Convert back to WebRTC format
    #         return av.VideoFrame.from_ndarray(processed_img, format="bgr24")  # Convert back to BGR


    # # Start Webcam Streaming
    # webrtc_streamer(
    #     key="age-detection",
    #     video_processor_factory=AgeDetectionProcessor,
    #     rtc_configuration=rtc_configuration,
    #     media_stream_constraints={"video": True, "audio": False},  # Enable video only
    # )

    st.write("📷 **Turn on your webcam to detect age in real-time**")

    # Start webcam button
    start_button = st.button("Start Webcam")

    if start_button:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video")
                break

            # Predict age for the current frame
            frame = model.predict_frame(frame)

            # Display the frame in Streamlit
            stframe.image(frame, channels="BGR", use_column_width=True)

        cap.release()
        cv2.destroyAllWindows()