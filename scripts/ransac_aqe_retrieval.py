import os
import json
import pickle
import math
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict
import sys

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,BASE_DIR)

from config import IMAGE_DIR,VOCAB_PATH,INDEX_PATH,PARSED_GT_PATH,FEATURES_DIR,KEYPOINTS_DIR
from config import SIFT_MAX_FEATURES,USE_ROOT_SIFT,TOP_N_PREFILTER,RANSAC_REPROJ_THRESHOLD,TOP_K_EXPAND,MIN_INLIERS_REQUIRED
from model.ROOTSIFT_feature import RootSIFTExtractor
from tools.evaluate_map import evaluate_system

# 引入底层的核心算子
from model.TFIDF_Engine import execute_bow_search,compute_query_weights
from model.RANSAC import pure_python_ransac_homography
from model.AQE import average_query_expansion  

def geometric_verification(query_kps, query_descs,db_kps,db_descs,ransac_thresh=RANSAC_REPROJ_THRESHOLD):
    """RANSAC 空间校验 (调用纯手写底层库)"""
    bf=cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
    matches=bf.match(query_descs,db_descs)
    
    # 单应性矩阵拥有8个自由度，数学底线是至少需要4对特征点
    if len(matches)<4: 
        return 0,[]
        
    # 注意：适配手写的NumPy矩阵运算，shape必须是 (-1, 2)
    src_pts=np.float32([query_kps[m.queryIdx] for m in matches]).reshape(-1,2)
    dst_pts=np.float32([db_kps[m.trainIdx] for m in matches]).reshape(-1,2)
    
    # 直接调用手写的RANSAC进行内点裁决
    inliers,inlier_indices=pure_python_ransac_homography(
        src_pts, 
        dst_pts, 
        threshold=ransac_thresh, 
        max_iters=1000
    )

    # 根据内点索引构建 inlier_pts 列表
    inlier_pts=[]
    for idx in inlier_indices:
        # 注意：这里 idx 是匹配对的索引，对应 matches[idx]
        qx,qy=query_kps[matches[idx].queryIdx]
        dx,dy=db_kps[matches[idx].trainIdx]
        inlier_pts.append({'query': [float(qx), float(qy)], 'db': [float(dx), float(dy)]})
    return inliers,inlier_pts

def execute_ransac_rerank(top_candidates,query_kps,query_descs,return_matches=False,ransac_thresh=5.0):
    """对候选列表执行 RANSAC 重排"""
    reranked_list=[]
    for db_img_name,bow_score in top_candidates:
        kp_path=os.path.join(KEYPOINTS_DIR,f"{db_img_name}.npy")
        desc_path=os.path.join(FEATURES_DIR,f"{db_img_name}.npy")
        
        inliers=0
        if os.path.exists(kp_path) and os.path.exists(desc_path):
            db_kps=np.load(kp_path)
            db_descs=np.load(desc_path)
            # 自动进入手写的RANSAC逻辑
            inliers,match_pts=geometric_verification(query_kps,query_descs,db_kps,db_descs,ransac_thresh)
            
        if return_matches:
            reranked_list.append((db_img_name,inliers,bow_score,match_pts))
        else:
            reranked_list.append((db_img_name,inliers,bow_score))
        
    # 按照内点数降序，倒排分数降序
    reranked_list.sort(key=lambda x: (x[1],x[2]),reverse=True)
    return reranked_list

def ransac_aqe_ultimate_retrieval(top_k_expand=TOP_K_EXPAND):
    print("加载引擎数据 (VOCAB, INDEX, GT)...")
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
    
    print(f"\n 启动终极架构: BoW -> 手写RANSAC校验 -> AQE扩展 -> BoW -> 手写RANSAC终审")
    for query_name,gt_info in tqdm(gt_data.items(),desc="全链路检索"):
        query_img_name=gt_info['query_img']
        bbox=gt_info['bbox']
        query_img_path=os.path.join(IMAGE_DIR,query_img_name)
        
        # 提取原始查询特征
        kps,descs=extractor.extract(query_img_path,bbox=bbox,multi_scale=True)
        if descs is None: 
            continue
        query_kps=np.array([kp.pt for kp in kps],dtype=np.float32)
        
        # TF-IDF权重计算，复用封装逻辑
        original_query_weights=compute_query_weights(descs,kmeans,idf)
            
        # 阶段1：初次BoW检索
        initial_candidates = execute_bow_search(original_query_weights, inverted_index, image_norms, TOP_N_PREFILTER)
        if not initial_candidates: 
            continue
            
        # 阶段2：第一次RANSAC
        ransac1_results = execute_ransac_rerank(initial_candidates, query_kps, descs)
        
        # 阶段3：安全的AQE扩展
        # 提取经过RANSAC校验的前K张图
        top_k_imgs_verified=[
            img for img,inl,score in ransac1_results[:top_k_expand]
            if inl>=MIN_INLIERS_REQUIRED # 内点数不足的线人直接丢弃
        ]
        
        expanded_weights=average_query_expansion(
            original_query_weights,
            top_k_imgs_verified,
            inverted_index,
            top_k_expand=top_k_expand
        )
                
        # 阶段4：扩展后的二次BoW检索
        expanded_candidates=execute_bow_search(expanded_weights,inverted_index,image_norms,TOP_N_PREFILTER)
        
        # 阶段5：RANSAC重排
        # 用扩展后得到的新候选名单，再和【原始查询图】比对一次空间结构
        final_ransac_results=execute_ransac_rerank(expanded_candidates,query_kps,descs)
        
        # 保存最终名次 (补上后缀)
        system_results[query_name]=[f"{img}.jpg" for img,inl,sc in final_ransac_results]
        
    print("\n正在计算终极mAP分数...")
    mAP_score=evaluate_system(PARSED_GT_PATH,system_results)
    
    print(f"RANSAC+AQE融合，最终mAP得分: {mAP_score:.4f}")


def search_single_query(query_img_path,bbox=None,top_k_expand=TOP_K_EXPAND,final_top_n=100,ransac_thresh=5.0):
    """
    单张图片检索接口
    """
    # 加载视觉词典
    with open(VOCAB_PATH,'rb') as f: 
        kmeans = pickle.load(f)

    # 加载倒排索引
    with open(INDEX_PATH,'rb') as f: 
        index_data = pickle.load(f)

    extractor=RootSIFTExtractor(max_features=SIFT_MAX_FEATURES,use_rootsift=USE_ROOT_SIFT)
    # 提取特征
    kps, descs = extractor.extract(query_img_path,bbox=bbox,multi_scale=True)
    if descs is None:
        return []
    
    query_kps=np.array([kp.pt for kp in kps],dtype=np.float32)
    
    # # 量化
    # words=kmeans.predict(descs)
    # query_tf=defaultdict(int)
    # for w in words:
    #     query_tf[w]+=1
    
    idf=index_data['idf']
    inverted_index=index_data['inverted_index']
    image_norms=index_data['image_norms']

    # 使用封装的权重计算
    original_query_weights=compute_query_weights(descs,kmeans,idf)
    
    # 初次检索
    initial_ranking=execute_bow_search(original_query_weights,inverted_index,image_norms,TOP_N_PREFILTER)
    if not initial_ranking:
        return []
    
    # RANSAC校验
    ransac1_results=execute_ransac_rerank(initial_ranking,query_kps,descs,return_matches=False,ransac_thresh=RANSAC_REPROJ_THRESHOLD)
    
    # AQE扩展
    top_k_imgs=[img for img,inl,score in ransac1_results[:top_k_expand]]
    expanded_weights=average_query_expansion(
        original_query_weights,
        top_k_imgs,
        inverted_index,
        top_k_expand=top_k_expand
    )
    
    # 二次检索
    final_ranking=execute_bow_search(expanded_weights,inverted_index,image_norms,TOP_N_PREFILTER)
    
    # 最终RANSAC
    final_results=execute_ransac_rerank(final_ranking,query_kps,descs,return_matches=True,ransac_thresh=ransac_thresh)
    
    return final_results[:final_top_n]

# if __name__ == '__main__':
#     ransac_aqe_ultimate_retrieval()