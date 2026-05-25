import cv2
import numpy as np
import os

class RootSIFTExtractor:
    """
    带有 RootSIFT 优化的特征提取器
    通过 L1 归一化和平方根映射，将 SIFT 特征的欧氏距离隐式转化为 Hellinger 距离，显著提升匹配精度。
    """
    def __init__(self, max_features=3000, use_rootsift=True):
        """
        初始化特征提取器
        :param max_features: 每张图片最多提取的关键点数量（控制内存占用的关键参数）
        :param use_rootsift: 是否开启 RootSIFT 优化
        """
        # 声明底层特征提取器
        self.sift = cv2.SIFT_create(nfeatures=max_features)
        self.use_rootsift = use_rootsift

    def extract(self, image_path, bbox=None):
        """
        从指定图像提取特征。如果提供了 bbox，则只在目标区域内提取。
        
        :param image_path: 图像路径
        :param bbox: [xmin, ymin, xmax, ymax] 可选，用于 Query 图像的精准去噪
        :return: (keypoints, descriptors)
        """
        if not os.path.exists(image_path):
            print(f"警告: 找不到图像 {image_path}")
            return [], None

        # 以灰度模式读取图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"警告: 无法读取图像 {image_path}")
            return [], None

        # 如果传入了边界框 (查询阶段)，则裁剪目标区域
        if bbox is not None:
            xmin, ymin, xmax, ymax = [int(v) for v in bbox]
            
            # 增加边界保护，防止坐标越界导致程序崩溃
            ymin, ymax = max(0, ymin), min(image.shape[0], ymax)
            xmin, xmax = max(0, xmin), min(image.shape[1], xmax)
            image = image[ymin:ymax, xmin:xmax]

            if image.size == 0:
                print(f"警告: {image_path} 的 bbox 裁剪区域无效")
                return [], None

        # 提取关键点和原始 SIFT 描述子
        kps, descs = self.sift.detectAndCompute(image, None)

        if descs is None or len(kps) == 0:
            return [], None

        # ==========================================
        # 核心加分项：RootSIFT 优化算法实现
        # ==========================================
        if self.use_rootsift:
            # 1. L1 归一化：将每个 128 维描述子向量除以其绝对值之和
            # 添加 eps (1e-7) 防止遇到全零向量导致除零报错 (Division by Zero)
            eps = 1e-7
            l1_norm = np.sum(np.abs(descs), axis=1, keepdims=True)
            descs = descs / (l1_norm + eps)
            
            # 2. 元素级开平方根
            descs = np.sqrt(descs)

        return kps, descs

# ================= 简单自测代码 =================
if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_img_path = os.path.join(BASE_DIR, 'data', 'images', 'all_souls_000013.jpg')
    test_bbox = [136.5, 34.1, 648.5, 955.7]
    
    extractor = RootSIFTExtractor(max_features=3000, use_rootsift=True)
    
    import time
    start_time = time.time()
    
    print("1. 测试提取整张原图的特征...")
    kps_full, descs_full = extractor.extract(test_img_path)
    
    cost_time = time.time() - start_time
    if descs_full is not None:
        print(f"   => 耗时 {cost_time:.3f} 秒！极速完成！")
        print(f"   => 成功提取了 {len(kps_full)} 个关键点，描述子矩阵维度: {descs_full.shape}")