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
        
    # 量化特征为视觉单词
    words=kmeans.predict(descs)
    
    # 统计查询图的 TF
    query_tf={}
    for w in words:
        query_tf[w]=query_tf.get(w,0)+1
        
    # 计算查询图的TF-IDF和向量长度
    query_norm_sq=0.0
    query_weights={}
    for w,tf in query_tf.items():
        weight=tf*idf[w]
        query_weights[w]=weight
        query_norm_sq+=weight**2
    query_norm=math.sqrt(query_norm_sq)
    
    if query_norm==0:
        return []

    # 利用倒排索引进行极速相似度计算
    scores=defaultdict(float)
    
    # 遍历查询图里的每一个单词
    for w,q_weight in query_weights.items():
        # 如果这个单词在数据库里存在
        if w in inverted_index:
            # 遍历包含这个单词的所有数据库图片
            for db_img,db_weight in inverted_index[w].items():
                # 累加点积
                scores[db_img]+=q_weight*db_weight
                
    # 余弦相似度归一化(除以查询图和数据库图的长度乘积)
    final_scores={}
    for db_img,dot_product in scores.items():
        db_norm=image_norms.get(db_img,1.0)
        final_scores[db_img]=dot_product/(query_norm*db_norm)
        
    # 按分数从高到低排序，截取Top-100作为召回结果
    ranked_list=sorted(final_scores.items(),key=lambda x:x[1],reverse=True)
    return [f"{img}.jpg" for img,score in ranked_list[:100]]

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

# if __name__ == '__main__':
#     run_baseline_evaluation()