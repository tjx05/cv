import os
import glob
import numpy as np
from tqdm import tqdm
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# 导入配置，确保参数和之前提特征时一模一样，保证一一对应
from config import IMAGE_DIR, KEYPOINTS_DIR, SIFT_MAX_FEATURES, USE_ROOT_SIFT
from model.ROOTSIFT_feature import RootSIFTExtractor

def extract_keypoints_only():
    os.makedirs(KEYPOINTS_DIR, exist_ok=True)
    image_paths = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
    extractor = RootSIFTExtractor(max_features=SIFT_MAX_FEATURES, use_rootsift=USE_ROOT_SIFT)
    
    for img_path in tqdm(image_paths, desc="提取并保存空间坐标 (x,y)"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(KEYPOINTS_DIR, f"{base_name}.npy")
        
        if os.path.exists(out_path):
            continue
            
        kps, _ = extractor.extract(img_path)
        if kps:
            # 提取每个特征点的 x, y 坐标
            pts = np.array([kp.pt for kp in kps], dtype=np.float32)
            np.save(out_path, pts)

if __name__ == '__main__':
    extract_keypoints_only()