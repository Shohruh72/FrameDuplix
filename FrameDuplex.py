from itertools import groupby
from pathlib import Path

import os
import cv2
import numpy as np
import torch
from scipy.spatial.distance import cosine
from skimage.metrics import structural_similarity as compare_ssim
from torchvision import models, transforms


# Base class for Frame Duplicate Detection
class FrameDet:
    def __init__(self, video_path, method_name):
        # Initialize with video path and detection method
        self.video_path = video_path
        self.method_name = method_name
        self.frames = self.extract_frames()  # Extract frames from video
        self.save_dir = Path(f"outputs/{self.method_name}_results")  # Output directory for results

    def save_frame(self, frame, frame_index):
        # Save a frame as an image file
        self.save_dir.mkdir(exist_ok=True)
        frame_path = self.save_dir / f'duplicate_frame_{frame_index}.jpg'
        cv2.imwrite(str(frame_path), frame)

    def save_video_without_duplicates(self, duplicates):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file {self.video_path}")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = None
        frame_index = 0

        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_filename = f'outputs/{base_name}_{self.method_name}.mp4'

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if out is None:
                height, width, _ = frame.shape
                out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
                if not out.isOpened():
                    print(f"Error: Unable to create video file {output_filename}")
                    break
            if frame_index not in duplicates:
                out.write(frame)
            frame_index += 1

        cap.release()
        if out:
            out.release()
            print(f"Video saved as {output_filename}")

    def extract_frames(self):
        # Extract and return frames from the video
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def detect_duplicates(self):
        # Placeholder for duplicate detection method
        raise NotImplementedError("This method should be implemented by subclasses.")

    @staticmethod
    def grayscale(frame):
        # Convert frame to grayscale
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# SSIM based duplicate detector
class SSIMDet(FrameDet):
    def __init__(self, video_path, ssim_threshold=0.995, mse_threshold=10):
        # Initialize with SSIM and MSE thresholds
        super().__init__(video_path, 'ssim')
        self.ssim_threshold = ssim_threshold
        self.mse_threshold = mse_threshold

    def calculate_frame_similarity(self, frame1, frame2):
        # Calculate SSIM and MSE between two frames
        ssim = compare_ssim(frame1, frame2)
        mse = np.mean((frame1.astype("double") - frame2.astype("double")) ** 2)
        return ssim, mse

    def detect_duplicates(self):
        # Detect duplicate frames based on SSIM and MSE
        duplicates = []
        previous_frame = None
        for i, frame in enumerate(self.frames):
            grayscale_frame = self.grayscale(frame)
            if previous_frame is not None:
                ssim, mse = self.calculate_frame_similarity(previous_frame, grayscale_frame)
                if ssim > self.ssim_threshold and mse < self.mse_threshold:
                    duplicates.append(i)
                    self.save_frame(frame, i)
            previous_frame = grayscale_frame
        return duplicates

    def detect_and_save(self):
        duplicates = self.detect_duplicates()
        self.save_video_without_duplicates(duplicates)


# Deep Learning based duplicate detector
class DLDet(FrameDet):
    def __init__(self, video_path, threshold=0.9):
        # Initialize with feature similarity threshold
        super().__init__(video_path, 'deep_learning')
        self.model = self.load_resnet50_model()  # Load ResNet50 pretrained model, you may change the model
        self.threshold = threshold

    @staticmethod
    def load_resnet50_model():
        # Load and return pretrained ResNet50 model
        model = models.resnet50(pretrained=True).eval()
        return model

    def preprocess_frame(self, frame):
        # Preprocess a frame for deep learning model
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return preprocess(frame)

    def extract_features(self, frame):
        # Extract features from a frame using the model
        frame_tensor = self.preprocess_frame(frame).unsqueeze(0)
        with torch.no_grad():
            return self.model(frame_tensor).view(1, -1)

    def detect_duplicates(self):
        # Detect duplicate frames based on deep learning features
        duplicate_frames = []
        prev_features = None
        for i, frame in enumerate(self.frames):
            features = self.extract_features(frame)
            if prev_features is not None and torch.norm(features - prev_features).item() < self.threshold:
                duplicate_frames.append(i)
                self.save_frame(frame, i)
            prev_features = features
        return duplicate_frames

    def detect_and_save(self):
        duplicates = self.detect_duplicates()
        self.save_video_without_duplicates(duplicates)


# Combined method using SSIM, Optical Flow, and Deep Learning
class ComboDet(FrameDet):
    def __init__(self, video_path, ssim_threshold=0.95, feature_similarity_threshold=0.1, flow_threshold=5.0):
        # Initialize with thresholds for SSIM, deep feature similarity, and optical flow
        super().__init__(video_path, 'combined')
        self.model = models.resnet50(pretrained=True).eval()
        self.ssim_threshold = ssim_threshold
        self.feature_similarity_threshold = feature_similarity_threshold
        self.flow_threshold = flow_threshold
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def compute_deep_features(self, frame):
        tensor = self.transform(frame).unsqueeze(0)
        with torch.no_grad():
            features = self.model(tensor)
        return features.view(-1)

    def calculate_optical_flow(self, prev_frame, current_frame):
        prev_gray = self.grayscale(prev_frame)
        gray = self.grayscale(current_frame)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow

    def detect_duplicates(self):
        duplicates = []
        prev_frame = None
        for i, frame in enumerate(self.frames):
            if prev_frame is not None:
                ssim, _ = compare_ssim(self.grayscale(prev_frame), self.grayscale(frame), full=True)
                flow = self.calculate_optical_flow(prev_frame, frame)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mean_mag = np.mean(mag)

                if ssim > self.ssim_threshold and mean_mag < self.flow_threshold:
                    deep_features_prev = self.compute_deep_features(prev_frame)
                    deep_features_current = self.compute_deep_features(frame)
                    deep_sim = 1 - cosine(deep_features_prev, deep_features_current)

                    if deep_sim > self.feature_similarity_threshold:
                        duplicates.append(i)
                        self.save_frame(frame, i)

            prev_frame = frame

        # Grouping consecutive frame indices
        grouped_duplicates = [list(g) for _, g in groupby(duplicates, key=lambda n, c=iter(duplicates): n - next(c))]
        return grouped_duplicates

    def detect_and_save(self):
        duplicates = self.detect_duplicates()
        self.save_video_without_duplicates(duplicates)


# Example Usage
def main():
    video_path = 'input/Clip1.mp4'

    # Select the method: 'ssim', 'deep_learning', or 'advanced'
    method = input("Select the method ('ssim' or 'deep_learning' or 'advanced'): ")

    if method == 'ssim':
        detector = SSIMDet(video_path)
    elif method == 'deep_learning':
        detector = DLDet(video_path)
    elif method == 'advanced':
        detector = ComboDet(video_path)  # combined approaches
    else:
        raise ValueError("Invalid method selected.")

    duplicates = detector.detect_duplicates()
    print(f"Duplicate frames using {method} method: {duplicates}")
    ''' "To optimize video saving without duplicates, enable the following feature; 
    however, note that disabling this may slightly increase processing time. '''
    # detector.detect_and_save()  # activate this lane to save video without duplicates


if __name__ == '__main__':
    main()
