import os
import glob
import numpy as np
from tqdm import tqdm
import sys

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from config import IMAGE_DIR,FEATURES_DIR,SIFT_MAX_FEATURES,USE_ROOT_SIFT
from model.ROOTSIFT_feature import RootSIFTExtractor

def extract_and_save_all_features(image_dir, output_dir):
    """
    遍历整个图像库，提取RootSIFT特征并保存为.npy文件
    """
    # 确保输出目录存在
    os.makedirs(output_dir,exist_ok=True)
    
    # 获取所有jpg图像的路径
    image_paths=glob.glob(os.path.join(image_dir,'*.jpg'))
    if len(image_paths) == 0:
        print(f"错误：在{image_dir}中找不到任何JPG图像，请检查路径！")
        return

    print(f"找到{len(image_paths)}张图像，准备开始特征提取...")
    
    # 初始化特征提取器 (采用标准配置：最多3000个点，开启RootSIFT)
    extractor=RootSIFTExtractor(max_features=SIFT_MAX_FEATURES,use_rootsift=USE_ROOT_SIFT)
    
    total_features=0
    failed_images=0

    # 使用tqdm包装循环，生成进度条
    for img_path in tqdm(image_paths,desc="提取特征中"):
        # 获取不带后缀的文件名，如'all_souls_000000'
        base_name=os.path.splitext(os.path.basename(img_path))[0]
        out_npy_path=os.path.join(output_dir, f"{base_name}.npy")
        
        # 断点续传机制：如果这个特征之前已经提取过了，直接跳过；这样即使程序跑到一半崩溃了，重新运行也不会从头开始。
        if os.path.exists(out_npy_path):
            continue
            
        # 提取特征
        kps,descs=extractor.extract(img_path)
        
        if descs is not None and len(descs) > 0:
            # 保存为numpy专用的高效二进制格式
            np.save(out_npy_path,descs)
            total_features+=len(descs)
        else:
            failed_images+=1

    print("\n全库特征提取完成")
    print(f"总计提取特征点数预估: 大于{total_features}个 (不含跳过的文件)")
    if failed_images >0:
        print(f"警告: 有{failed_images}张图像提取失败或未找到特征点。")

if __name__=='__main__':
    # 执行批处理
    print(f"当前配置->SIFT上限: {SIFT_MAX_FEATURES}, RootSIFT: {USE_ROOT_SIFT}")
    extract_and_save_all_features(IMAGE_DIR,FEATURES_DIR)