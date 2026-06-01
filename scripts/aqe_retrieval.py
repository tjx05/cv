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

# 增加导入 TOP_N_PREFILTER 以适配 execute_bow_search 的参数
from config import IMAGE_DIR,VOCAB_PATH,INDEX_PATH,PARSED_GT_PATH,SIFT_MAX_FEATURES,USE_ROOT_SIFT, TOP_N_PREFILTER
from model.ROOTSIFT_feature import RootSIFTExtractor
from tools.evaluate_map import evaluate_system

# 导入底层的核心算子
from model.TFIDF_Engine import execute_bow_search, compute_query_weights
from model.AQE import average_query_expansion


def aqe_retrieval(top_k_expand=5):
    print("加载词典与倒排索引……")
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
    
    print(f"\n开始 AQE 查询扩展评测(扩展前{top_k_expand}张图的特征)……")
    for query_name,gt_info in tqdm(gt_data.items(),desc="AQE检索中"):
        query_img_name=gt_info['query_img']
        bbox=gt_info['bbox']
        query_img_path=os.path.join(IMAGE_DIR,query_img_name)
        
        # 提取原始查询特征
        kps,descs=extractor.extract(query_img_path,bbox=bbox)
        if descs is None: 
            continue
            
        # 调用封装函数计算原始查询的 TF-IDF 权重向量
        original_query_weights = compute_query_weights(descs, kmeans, idf)
            
        # 第一次检索，调用统一的倒排查询接口
        initial_ranking = execute_bow_search(original_query_weights, inverted_index, image_norms, TOP_N_PREFILTER)
        
        # 如果什么都没搜到，直接跳过
        if not initial_ranking:
            system_results[query_name]=[]
            continue
            
        # AQE 核心逻辑: 提取第一次检索排名前K的图片名（即盲目信任它们是对的）
        top_k_imgs=[img for img,score in initial_ranking[:top_k_expand]]
        
        #直接调用独立封装好的 AQE 聚合算法
        expanded_weights = average_query_expansion(
            original_query_weights,
            top_k_imgs,
            inverted_index,
            top_k_expand=top_k_expand
        )
                
        # 第二次检索，由于最终只存前 100 名，这里直接让引擎返回 top_n=100
        final_ranking = execute_bow_search(expanded_weights, inverted_index, image_norms, top_n=100)
        
        # 保存重排后的纯图片名列表
        system_results[query_name]=[f"{img}.jpg" for img,score in final_ranking]
        
    print("\n正在计算mAP分数……")
    mAP_score=evaluate_system(PARSED_GT_PATH,system_results)
    
    print(f"AQE扩展系统构建完毕，最终mAP得分:{mAP_score:.4f}")

if __name__ == '__main__':
    aqe_retrieval(top_k_expand=5)