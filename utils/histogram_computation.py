from skimage.feature import hog
from skimage.color import rgb2gray
from utils.constants import *
import cv2
from pathlib import Path
import pandas as pd
import numpy as np


def create_feature_sets_with_hog(train_df_color, test_df_color, train_df, test_df):
    """
    Combine color histograms, ShapeId, and HOG features.
    """
    X_color_train = np.stack(train_df_color['color_feature_vector'].values)
    X_color_test = np.stack(test_df_color['color_feature_vector'].values)
    
    # shape ID removed from training
    #shape_train = pd.get_dummies(train_df_color['ShapeId'], prefix='Shape')
    #shape_test = pd.get_dummies(test_df_color['ShapeId'], prefix='Shape')
    #shape_train, shape_test = shape_train.align(shape_test, join='outer', axis=1, fill_value=0)
    
    X_hog_train, X_hog_test = [], []
    for df, output_list in zip([train_df, test_df], [X_hog_train, X_hog_test]):
        for idx, row in df.iterrows():
            img_path = Path(DF_PATH) / row['Path']
            img = cv2.imread(str(img_path))
            # zeroes if no image
            if img is None:
                output_list.append(np.zeros(324))  
                continue
            x1, y1, x2, y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
            roi = img[y1:y2, x1:x2]
            # Resize ROI to fixed size
            roi_resized = cv2.resize(roi, (64, 64))
            hog_feat = extract_hog_features(roi_resized)
            output_list.append(hog_feat)
    
    X_hog_train = np.array(X_hog_train)
    X_hog_test = np.array(X_hog_test)
    
    # stacking features separately
    X_train = np.hstack([X_color_train, X_hog_train])
    X_test = np.hstack([X_color_test, X_hog_test])
    
    y_train = train_df['ClassId'].values
    y_test = test_df['ClassId'].values
    
    return X_train, X_test, y_train, y_test





def extract_hog_features(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    """
    Extract HOG features from an image.
    
    Args:y
        image: RGB or BGR image (numpy array)
        pixels_per_cell: Size of the cell
        cells_per_block: Number of cells per block
        orientations: Number of orientation bins
    
    Returns:
        HOG feature vector
    """
    # Convert to grayscale
    gray = rgb2gray(image) if image.shape[2] == 3 else image
    
    features = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )
    
    return features

def extract_histograms_from_dataset(df, max_samples=None):
    """
    Extract color histograms for all ROIs in your dataset
    """
    features_list = []
    failed_paths = []
    
    # Limit samples ias df.head(x)
    if max_samples:
        df = df.head(max_samples)
    
    for idx, row in df.iterrows():
        try:
            img_path = Path(DF_PATH) / row['Path']
            
            if not img_path.exists():
                print(f"Image not found: {img_path}")
                failed_paths.append(row['Path'])
                continue
            
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Could not read image: {img_path}")
                failed_paths.append(row['Path'])
                continue
            
            # Extract ROI using your bounding box coordinates
            # Assuming Roi.X1, Roi.Y1, Roi.X2, Roi.Y2 are the coordinates
            x1, y1, x2, y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
            
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                print(f"Invalid ROI coordinates in row {idx}")
                continue
            
            
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                print(f"Empty ROI in row {idx}")
                continue
        
            histograms = calculate_comprehensive_histograms(roi)
            
            # feature row
            feature_row = {
                'index': idx,
                'ClassId': row['ClassId'],
                'SignId': row['SignId'],
                'ColorId': row['ColorId'],
                'ShapeId': row['ShapeId'],
                'Path': row['Path']
            }
            
            
            feature_row.update(histograms)
            features_list.append(feature_row)
            
            if idx % 1000 == 0:
                print(f"Processed {idx} samples...")
                
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            failed_paths.append(row['Path'])
    
    print(f"Successfully processed {len(features_list)} samples")
    print(f"Failed to process {len(failed_paths)} samples")
    
    return pd.DataFrame(features_list), failed_paths

def calculate_comprehensive_histograms(roi, bins=32):
    """
    Calculate comprehensive color histograms from ROI
    """
    # Convert to different color spaces
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    
    histograms = {}
    
    # BGR histograms
    for i, color in enumerate(['B', 'G', 'R']):
        hist = cv2.calcHist([roi], [i], None, [bins], [0, 256])
        histograms[f'hist_bgr_{color}'] = cv2.normalize(hist, hist).flatten()
    
    # HSV histograms
    for i, color in enumerate(['H', 'S', 'V']):
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256])
        histograms[f'hist_hsv_{color}'] = cv2.normalize(hist, hist).flatten()
    
    # LAB histograms
    for i, color in enumerate(['L', 'A', 'B']):
        hist = cv2.calcHist([lab], [i], None, [bins], [0, 256])
        histograms[f'hist_lab_{color}'] = cv2.normalize(hist, hist).flatten()
    
    # Combine
    all_features = []
    for key in sorted(histograms.keys()):
        all_features.extend(histograms[key])
    
    histograms['color_feature_vector'] = np.array(all_features)
    
    return histograms
