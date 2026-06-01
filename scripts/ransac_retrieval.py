import os
import json
import pickle
import numpy as np
import cv2
from tqdm import tqdm
import sys

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,BASE_DIR)

from config import IMAGE_DIR,VOCAB_PATH,INDEX_PATH,PARSED_GT_PATH,FEATURES_DIR,KEYPOINTS_DIR
from config import SIFT_MAX_FEATURES,USE_ROOT_SIFT,TOP_N_PREFILTER,RANSAC_REPROJ_THRESHOLD,TOP_K_EXPAND
from model.ROOTSIFT_feature import RootSIFTExtractor
from tools.evaluate_map import evaluate_system

# 倒排查询逻辑
from model.TFIDF_Engine import compute_query_weights, execute_bow_search

from model.RANSAC import pure_python_ransac_homography

def geometric_verification(query_kps,query_descs,db_kps,db_descs):
    """
    RANSAC核心逻辑：计算查询图和数据库图的内点数量
    """
    # 暴力匹配描述子 (使用欧氏距离，开启交叉验证寻找最佳匹配)
    bf=cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
    matches=bf.match(query_descs,db_descs)
    
    # 算单应性矩阵至少需要4对点
    if len(matches)<4:
        return 0
        
    # NumPy 矩阵运算，shape 必须是 (-1, 2)
    src_pts=np.float32([query_kps[m.queryIdx] for m in matches]).reshape(-1, 2)
    dst_pts=np.float32([db_kps[m.trainIdx] for m in matches]).reshape(-1, 2)
    
    # 调用手写的 RANSAC 进行内点裁决
    inliers = pure_python_ransac_homography(
        src_pts, 
        dst_pts, 
        threshold=RANSAC_REPROJ_THRESHOLD, 
        max_iters=1000
    )
    
    return inliers

def ransac_retrieval():
    print("加载词典与倒排索引...")
    with open(VOCAB_PATH,'rb') as f: 
        kmeans=pickle.load(f)
    with open(INDEX_PATH,'rb') as f: 
        index_data=pickle.load(f)
    with open(PARSED_GT_PATH,'r',encoding='utf-8') as f: 
        gt_data=json.load(f)
        
    extractor=RootSIFTExtractor(max_features=SIFT_MAX_FEATURES,use_rootsift=USE_ROOT_SIFT)
    inverted_index=index_data['inverted_index']
    idf=index_data['idf']
    image_norms=index_data['image_norms']
    
    system_results={}
    
    print(f"\n开始RANSAC空间重排评测(先取Top-{TOP_N_PREFILTER}初筛)...")
    for query_name,gt_info in tqdm(gt_data.items(),desc="RANSAC检索中"):
        query_img_name=gt_info['query_img']
        bbox=gt_info['bbox']
        query_img_path=os.path.join(IMAGE_DIR,query_img_name)
        
        # 查询扩展与特征提取
        kps,descs=extractor.extract(query_img_path,bbox=bbox)
        if descs is None: 
            continue
            
        # 查询坐标点：直接从提取出的kps里拿
        query_kps=np.array([kp.pt for kp in kps],dtype=np.float32)
        
        # TF-IDF 权重计算
        query_weights = compute_query_weights(descs, kmeans, idf)
        
        # 倒排极速初筛
        top_n_candidates = execute_bow_search(query_weights, inverted_index, image_norms, TOP_N_PREFILTER)
        
        # RANSAC重排序
        reranked_list=[]
        for db_img_name,bow_score in top_n_candidates:
            # 读取离线保存的坐标和特征
            kp_path=os.path.join(KEYPOINTS_DIR,f"{db_img_name}.npy")
            desc_path=os.path.join(FEATURES_DIR,f"{db_img_name}.npy")
            
            if os.path.exists(kp_path) and os.path.exists(desc_path):
                db_kps=np.load(kp_path)
                db_descs=np.load(desc_path)
                
                # 执行空间校验
                inliers=geometric_verification(query_kps,descs,db_kps,db_descs)
            else:
                inliers=0
                
            reranked_list.append({
                'img_name': f"{db_img_name}.jpg",
                'inliers': inliers,
                'bow_score': bow_score
            })
            
        # 优先按照内点数量降序，内点相同的按照原始倒排分数降序
        reranked_list.sort(key=lambda x: (x['inliers'],x['bow_score']),reverse=True)
        
        # 提取重排后的纯图片名列表
        system_results[query_name]=[item['img_name'] for item in reranked_list]
        
    print("\n正在计算 mAP 分数...")
    mAP_score=evaluate_system(PARSED_GT_PATH,system_results)
    
    print(f"RANSAC系统构建完毕，最终mAP得分: {mAP_score:.4f}")

if __name__=='__main__':
    ransac_retrieval()