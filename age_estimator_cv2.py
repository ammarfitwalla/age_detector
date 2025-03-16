import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
import cv2 as cv
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
        """Add a labeled bounding box to the image."""
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv.rectangle(image, p1, p2, color, thickness=lw, lineType=cv.LINE_AA)
        if label:
            tf = max(lw - 1, 1)
            w, h = cv.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
            outside = p1[1] - h - 3 >= 0
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv.rectangle(image, p1, p2, color, -1, cv.LINE_AA)
            cv.putText(
                image,
                label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv.LINE_AA,
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

        for i, box in enumerate(bboxes):
            box = np.clip(box, 0, np.inf).astype(np.uint32)
            label = f"{'Man' if genders[i] == 0 else 'Woman'}: {ages[i].item()} years old"

            # Check if the person is below 18 and add "Below 18" text
            if ages[i].item() < 18:
                label += " (Below 18)"
                cv.putText(
                    ndarray_image,
                    "Below 18",
                    (box[0], box[1] - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),  # Red color
                    2,
                    cv.LINE_AA,
                )

            self.plot_box_and_label(
                ndarray_image,
                max(ndarray_image.shape) // 400,
                box,
                label,
                color=(255, 0, 0),
            )

        return ndarray_image

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

        for i, box in enumerate(bboxes):
            box = np.clip(box, 0, np.inf).astype(np.uint32)
            thickness = max(image_shape) // 400
            thickness = int(max(np.ceil(thickness), 1))
            label = f"{'Man' if genders[i] == 0 else 'Woman'}: {ages[i].item()} years old"

            # Check if the person is below 18 and add "Below 18" text
            if ages[i].item() < 18:
                label += " (Below 18)"
                cv.putText(
                    ndarray_image,
                    "Below 18",
                    (box[0], box[1] - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),  # Red color
                    2,
                    cv.LINE_AA,
                )

            self.plot_box_and_label(
                ndarray_image, thickness, box, label, color=(255, 0, 0)
            )

        return ndarray_image
