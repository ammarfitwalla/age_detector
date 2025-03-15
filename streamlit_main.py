import streamlit as st
import cv2
from PIL import Image
import tempfile
import gdown
import os

# Import your AgeEstimator class
from age_estimator import AgeEstimator  # Ensure 'your_script' is the filename containing the AgeEstimator class


def download_weights():
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
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Convert uploaded image to a format suitable for prediction
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)

        # Predict
        st.write("⏳ Processing Image...")
        processed_image = model.predict(temp_file.name)

        # Display result
        st.image(processed_image, caption="Predicted Image", use_column_width=True)
        st.success("✅ Prediction Completed!")

elif option == "Live Webcam":
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
            stframe.image(frame, channels="RGB", use_column_width=True)

        cap.release()
        cv2.destroyAllWindows()
