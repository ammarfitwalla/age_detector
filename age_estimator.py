import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T
from AgeNet.models import Model
from Facenet.models.mtcnn import MTCNN


class AgeEstimator:
    def __init__(self, face_size=64, weights=None, device="cpu", tpx=500):
        self.thickness_per_pixels = tpx
        self.face_size = (
            (face_size, face_size) if isinstance(face_size, int) else face_size
        )
        self.device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )

        # Initialize models
        self.facenet_model = MTCNN(device=self.device)
        self.model = Model().to(self.device)
        self.model.eval()

        # Load weights if provided
        if weights:
            self.model.load_state_dict(torch.load(weights, map_location="cpu"))
            print(f"Weights loaded successfully from path: {weights}")
            print("=" * 60)

    def transform(self, image):
        """Transform input face image for the model."""
        return T.Compose(
            [
                T.Resize(self.face_size),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )(image)

    @staticmethod
    def plot_box_and_label(
        image, lw, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)
    ):
        """Add a labeled bounding box to the image using Pillow."""
        draw = ImageDraw.Draw(image)
        
        # Calculate font size dynamically based on image resolution
        image_width, image_height = image.size
        base_font_size = max(image_width, image_height) // 50  # Adjust divisor for scaling
        try:
            font = ImageFont.truetype("arial.ttf", size=base_font_size)  # Use a system font
        except IOError:
            font = ImageFont.load_default()  # Fallback to default font if "arial.ttf" is not available

        # Draw bounding box
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        draw.rectangle([p1, p2], outline=color, width=lw)

        # Draw label inside the bounding box
        if label:
            # Calculate text size
            text_width, text_height = draw.textsize(label, font=font)
            
            # Position the text inside the bounding box (top-left corner)
            text_position = (p1[0] + 5, p1[1] + 5)  # Add a small offset (5 pixels) from the top-left corner
            
            # Draw a background rectangle for the text
            draw.rectangle(
                [text_position, (text_position[0] + text_width, text_position[1] + text_height)],
                fill=color,
            )
            
            # Draw the text
            draw.text(
                text_position,
                label,
                fill=txt_color,
                font=font,
            )

    def padding_face(self, box, padding=10):
        """Apply padding to bounding box."""
        return [box[0] - padding, box[1] - padding, box[2] + padding, box[3] + padding]

    def predict_frame(self, frame, min_prob=0.9):
        """Process a single video frame for real-time predictions."""
        image = Image.fromarray(frame)
        ndarray_image = np.array(frame)
        bboxes, prob = self.facenet_model.detect(image)

        if bboxes is None:
            return ndarray_image

        bboxes = bboxes[prob > min_prob]
        face_images = []
        for box in bboxes:
            box = np.clip(box, 0, np.inf).astype(np.uint32)
            face = image.crop(box)
            transformed_face = self.transform(face)
            face_images.append(transformed_face)

        if not face_images:
            return ndarray_image

        face_images = torch.stack(face_images, dim=0)
        genders, ages = self.model(face_images)
        genders = torch.round(genders)
        ages = torch.round(ages).long()

        # Check if any face is below 18 years old
        below_18 = any(age < 18 for age in ages)
        if below_18:
            # Add a visual alert to the image
            draw = ImageDraw.Draw(image)
            alert_text = "Alert: Person below 18 years old detected!"
            font = ImageFont.load_default()
            text_width, text_height = draw.textsize(alert_text, font=font)
            draw.rectangle(
                [(10, 10), (10 + text_width + 10, 10 + text_height + 10)],
                fill=(255, 0, 0),  # Red background
            )
            draw.text(
                (15, 15),
                alert_text,
                fill=(255, 255, 255),  # White text
                font=font,
            )

        for i, box in enumerate(bboxes):
            box = np.clip(box, 0, np.inf).astype(np.uint32)
            label = (
                f"{'Man' if genders[i] == 0 else 'Woman'}: {ages[i].item()} years old"
            )
            self.plot_box_and_label(
                image,
                max(ndarray_image.shape) // 400,
                box,
                label,
                color=(255, 0, 0),
            )

        return np.array(image)

    def predict(self, img_path, min_prob=0.9):
        """Process an image file for predictions."""
        image = Image.open(img_path)
        ndarray_image = np.array(image)
        image_shape = ndarray_image.shape
        bboxes, prob = self.facenet_model.detect(image)
        if bboxes is None:
            return ndarray_image

        bboxes = bboxes[prob > min_prob]
        face_images = []
        for box in bboxes:
            box = np.clip(box, 0, np.inf).astype(np.uint32)
            padding = max(image_shape) * 5 / self.thickness_per_pixels
            padding = int(max(padding, 10))
            box = self.padding_face(box, padding)
            face = image.crop(box)
            transformed_face = self.transform(face)
            face_images.append(transformed_face)

        if not face_images:
            return ndarray_image

        face_images = torch.stack(face_images, dim=0)
        genders, ages = self.model(face_images)
        genders = torch.round(genders)
        ages = torch.round(ages).long()

        # Check if any face is below 18 years old
        below_18 = any(age < 18 for age in ages)
        if below_18:
            # Add a visual alert to the image
            draw = ImageDraw.Draw(image)
            alert_text = "Alert: Person below 18 years old detected!"
            font = ImageFont.load_default()
            text_width, text_height = draw.textsize(alert_text, font=font)
            draw.rectangle(
                [(10, 10), (10 + text_width + 10, 10 + text_height + 10)],
                fill=(255, 0, 0),  # Red background
            )
            draw.text(
                (15, 15),
                alert_text,
                fill=(255, 255, 255),  # White text
                font=font,
            )

        for i, box in enumerate(bboxes):
            box = np.clip(box, 0, np.inf).astype(np.uint32)
            thickness = max(image_shape) // 400
            thickness = int(max(np.ceil(thickness), 1))
            label = (
                f"{'Man' if genders[i] == 0 else 'Woman'}: {ages[i].item()} years old"
            )
            self.plot_box_and_label(
                image, thickness, box, label, color=(255, 0, 0)
            )

        return np.array(image)
