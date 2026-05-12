# backend/services/preprocessing/image_preprocess.py

import numpy as np
import cv2
from pathlib import Path
from services.preprocessing.train_test_split import split_dataset

class ImagePreprocessor:
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size

    def run(self, session: dict):
        """
        Expects session['dataset_path'] to be a directory where 
        subfolders are class names (standard ImageNet style).
        """
        data_path = Path(session["dataset_path"])
        images = []
        labels = []
        class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])

        for idx, label in enumerate(class_names):
            class_dir = data_path / label
            for img_path in class_dir.glob("*.[jJ][pP][gG]"):
                # Load and Resize
                img = cv2.imread(str(img_path))
                if img is None: continue
                img = cv2.resize(img, self.target_size)
                img = img / 255.0  # Normalization
                
                images.append(img.flatten()) # Flatten for Layer 5 classical models
                labels.append(idx)

        # Convert to arrays
        X = np.array(images)
        y = np.array(labels)

        # Use your existing split utility
        # Note: We pass a dummy DF because split_dataset expects one
        import pandas as pd
        df = pd.DataFrame(X)
        df['__target__'] = y

        return split_dataset(
            df=df,
            target_column='__target__',
            train_split=session.get("split_ratio", 0.8),
            problem_type="classification"
        )