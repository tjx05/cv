import os

# 全局路径配置
# 自动获取项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据集路径
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')               # 存放 5062 张原图
GT_DIR = os.path.join(DATA_DIR, 'gt_files_170407')         # 存放官方 txt 标注
PARSED_GT_PATH = os.path.join(DATA_DIR, 'parsed_groundtruth.json') # 预处理生成的 JSON

# 输出文件路径
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
KEYPOINTS_DIR = os.path.join(OUTPUT_DIR, 'keypoints')        # 存放特征点 (x,y) 坐标
FEATURES_DIR = os.path.join(OUTPUT_DIR, 'features')        # 存放 .npy 离线特征
VOCAB_PATH = os.path.join(OUTPUT_DIR, 'vocab.pkl')         # 视觉词典模型
INDEX_PATH = os.path.join(OUTPUT_DIR, 'inverted_index.pkl')# 倒排索引表


# 特征提取参数
SIFT_MAX_FEATURES = 3000      # 每张图最多提取的特征点数量 (防内存溢出)
USE_ROOT_SIFT = True          # 是否开启 RootSIFT

# 词典与聚类参数 (Vocabulary & KMeans)
VOCAB_SIZE_K = 50000         # 视觉词典大小 K 
KMEANS_BATCH_SIZE = 50000     # 小批量 K-Means 的 Batch 大小
KMEANS_SAMPLE_RATE = 0.1      # 聚类时从 1200 万特征中随机采样的比例 (防内存溢出)


# 检索与重排参数
TOP_N_PREFILTER = 300         # 倒排索引初筛保留的前 N 张候选图
RANSAC_REPROJ_THRESHOLD = 5.0 # RANSAC 内点容差阈值 (像素)
MIN_INLIERS_REQUIRED = 10     # 认定为成功匹配的最少内点数

TOP_K_EXPAND = 5              # 参与 AQE 特征融合的最高置信度图像数量