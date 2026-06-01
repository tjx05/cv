import os
import json
import pickle
import math
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import sys

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,BASE_DIR)

from config import IMAGE_DIR,VOCAB_PATH,INDEX_PATH,PARSED_GT_PATH,SIFT_MAX_FEATURES,USE_ROOT_SIFT
from model.ROOTSIFT_feature import RootSIFTExtractor

from model.TFIDF_Engine import compute_query_weights, execute_bow_search

# 导入评测工具
try:
    from tools.evaluate_map import evaluate_system
except ImportError:
    print("找不到 tools/evaluate_map.py")
    sys.exit(1)

def search_image(query_img_path,bbox,extractor,kmeans,index_data):
    """
    对单张查询图像进行检索，返回计算好得分的图片列表
    """
    inverted_index=index_data['inverted_index']
    idf=index_data['idf']
    image_norms=index_data['image_norms']
    
    # 提取查询图像特征(使用bbox剔除背景)
    kps, descs=extractor.extract(query_img_path,bbox=bbox)
    if descs is None or len(descs)==0:
        return []
        
    #计算 TF-IDF 权重
    query_weights = compute_query_weights(descs, kmeans, idf)

    # 完成倒排极速检索
    ranked_list = execute_bow_search(query_weights, inverted_index, image_norms, top_n=100)
        
    # 按分数从高到低排序，截取Top-100作为召回结果 (execute_bow_search已排好序)
    return [f"{img}.jpg" for img, score in ranked_list]


def run_baseline_evaluation():
    print("加载词典与倒排索引……")
    with open(VOCAB_PATH,'rb') as f:
        kmeans=pickle.load(f)
    with open(INDEX_PATH,'rb') as f:
        index_data=pickle.load(f)
        
    print("加载Ground Truth数据……")
    with open(PARSED_GT_PATH,'r',encoding='utf-8') as f:
        gt_data=json.load(f)
        
    extractor=RootSIFTExtractor(max_features=SIFT_MAX_FEATURES,use_rootsift=USE_ROOT_SIFT)
    
    # 用于存储系统的所有预测结果
    system_results={}
    
    print("\n开始全库检索与评测(共55个Query)……")
    for query_name,gt_info in tqdm(gt_data.items(),desc="检索进度"):
        query_img_name=gt_info['query_img']
        bbox=gt_info['bbox']
        query_img_path=os.path.join(IMAGE_DIR,query_img_name)
        
        # 执行检索
        ranked_results=search_image(query_img_path,bbox,extractor,kmeans, index_data)
        system_results[query_name]=ranked_results
        
    print("正在计算mAP分数……")
    mAP_score=evaluate_system(PARSED_GT_PATH,system_results)
    
    print(f"Baseline系统构建完毕，最终mAP得分: {mAP_score:.4f}")

if __name__ == '__main__':
    run_baseline_evaluation()