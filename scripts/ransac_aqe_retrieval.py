import os
import json
import pickle
import math
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import IMAGE_DIR, VOCAB_PATH, INDEX_PATH, PARSED_GT_PATH, FEATURES_DIR, KEYPOINTS_DIR
from config import SIFT_MAX_FEATURES, USE_ROOT_SIFT, TOP_N_PREFILTER, RANSAC_REPROJ_THRESHOLD,TOP_K_EXPAND
from model.ROOTSIFT_feature import RootSIFTExtractor
from tools.evaluate_map import evaluate_system

# 👉 [核心修改点 1]：引入底层的核心算子，复用既有逻辑，保持代码 DRY
from model.TFIDF_Engine import execute_bow_search, compute_query_weights
from model.RANSAC import pure_python_ransac_homography
from model.AQE import average_query_expansion  

def geometric_verification(query_kps, query_descs, db_kps, db_descs):
    """RANSAC 空间校验 (调用纯手写底层库)"""
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(query_descs, db_descs)
    
    # 单应性矩阵拥有8个自由度，数学底线是至少需要 4 对特征点
    if len(matches) < 4: 
        return 0
        
    # 注意：适配手写的 NumPy 矩阵运算，shape 必须是 (-1, 2)
    src_pts = np.float32([query_kps[m.queryIdx] for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([db_kps[m.trainIdx] for m in matches]).reshape(-1, 2)
    
    # 直接调用手写的 RANSAC 进行内点裁决
    inliers = pure_python_ransac_homography(
        src_pts, 
        dst_pts, 
        threshold=RANSAC_REPROJ_THRESHOLD, 
        max_iters=1000
    )
    return inliers

def execute_ransac_rerank(top_candidates, query_kps, query_descs):
    """对候选列表执行 RANSAC 重排"""
    reranked_list = []
    for db_img_name, bow_score in top_candidates:
        kp_path = os.path.join(KEYPOINTS_DIR, f"{db_img_name}.npy")
        desc_path = os.path.join(FEATURES_DIR, f"{db_img_name}.npy")
        
        inliers = 0
        if os.path.exists(kp_path) and os.path.exists(desc_path):
            db_kps = np.load(kp_path)
            db_descs = np.load(desc_path)
            # 这里会自动进入你手写的 RANSAC 逻辑
            inliers = geometric_verification(query_kps, query_descs, db_kps, db_descs)
            
        reranked_list.append((db_img_name, inliers, bow_score))
        
    # 按照内点数降序，倒排分数降序
    reranked_list.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return reranked_list


def ransac_aqe_ultimate_retrieval(top_k_expand=5):
    print("📖 加载引擎数据 (VOCAB, INDEX, GT)...")
    with open(VOCAB_PATH, 'rb') as f: kmeans = pickle.load(f)
    with open(INDEX_PATH, 'rb') as f: index_data = pickle.load(f)
    with open(PARSED_GT_PATH, 'r', encoding='utf-8') as f: gt_data = json.load(f)
        
    extractor = RootSIFTExtractor(max_features=SIFT_MAX_FEATURES, use_rootsift=USE_ROOT_SIFT)
    inverted_index = index_data['inverted_index']
    idf = index_data['idf']
    image_norms = index_data['image_norms']
    
    system_results = {}
    
    print(f"\n🚀 启动终极架构: BoW -> 手写RANSAC校验 -> AQE扩展 -> BoW -> 手写RANSAC终审")
    for query_name, gt_info in tqdm(gt_data.items(), desc="全链路检索"):
        query_img_name = gt_info['query_img']
        bbox = gt_info['bbox']
        query_img_path = os.path.join(IMAGE_DIR, query_img_name)
        
        # 开启多尺度 (1.0 和 0.8) 提取特征
        kps, descs = extractor.extract(query_img_path, bbox=bbox, multi_scale=True)
        if descs is None: continue
        query_kps = np.array([kp.pt for kp in kps], dtype=np.float32)
        
        # 👉 [核心修改点 3]：一行代码搞定 TF-IDF 权重计算，复用封装逻辑
        original_query_weights = compute_query_weights(descs, kmeans, idf)
            
        # ==========================================
        # 阶段 1: 初次 BoW 检索
        # ==========================================
        initial_candidates = execute_bow_search(original_query_weights, inverted_index, image_norms, TOP_N_PREFILTER)
        if not initial_candidates: continue
            
        # ==========================================
        # 阶段 2: 第一次 手写 RANSAC (寻找绝对正确的“线人”)
        # ==========================================
        ransac1_results = execute_ransac_rerank(initial_candidates, query_kps, descs)
        
        # ==========================================
        # 阶段 3: 调用独立 AQE 模块进行查询扩展
        # ==========================================
        top_k_imgs_verified = [img for img, inl, score in ransac1_results[:TOP_K_EXPAND]]
        
        expanded_weights = average_query_expansion(
            original_query_weights,
            top_k_imgs_verified,
            inverted_index,
            top_k_expand=TOP_K_EXPAND 
        )
                
        # ==========================================
        # 阶段 4: 扩展后的二次 BoW 检索
        # ==========================================
        expanded_candidates = execute_bow_search(expanded_weights, inverted_index, image_norms, TOP_N_PREFILTER)
        
        # ==========================================
        # 阶段 5: 终极 手写 RANSAC 重排
        # ==========================================
        # 必须用扩展后得到的新候选名单，再和【原始查询图】比对一次空间结构
        final_ransac_results = execute_ransac_rerank(expanded_candidates, query_kps, descs)
        
        # 保存最终名次 (补上后缀)
        system_results[query_name] = [f"{img}.jpg" for img, inl, sc in final_ransac_results]
        
    print("\n📊 正在计算终极 mAP 分数...")
    mAP_score = evaluate_system(PARSED_GT_PATH, system_results)
    
    print("==================================================")
    print(f"🏆 终局之战！纯手写 RANSAC + AQE 完美融合，最终 mAP 得分: {mAP_score:.4f}")
    print("==================================================")

if __name__ == '__main__':
    ransac_aqe_ultimate_retrieval()