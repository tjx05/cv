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
from tools.evaluate_map import evaluate_system

def execute_search(query_weights,inverted_index,image_norms):
    """
    基础的倒排索引查询函数：传入权重向量，返回按分数排序的图片列表
    """
    scores=defaultdict(float)
    query_norm_sq=sum(w**2 for w in query_weights.values())
    query_norm=math.sqrt(query_norm_sq)
    
    if query_norm==0:
        return []

    # 余弦相似度分子：点积累加
    for w,q_weight in query_weights.items():
        if w in inverted_index:
            for db_img,db_weight in inverted_index[w].items():
                scores[db_img]+=q_weight*db_weight
                
    # 余弦相似度分母：长度归一化
    final_scores={}
    for db_img,dot_product in scores.items():
        db_norm=image_norms.get(db_img,1.0)
        final_scores[db_img]=dot_product/(query_norm*db_norm)
        
    # 按分数降序排序
    return sorted(final_scores.items(),key=lambda x:x[1],reverse=True)

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
            
        words=kmeans.predict(descs)
        query_tf=defaultdict(int)
        for w in words: 
            query_tf[w]+=1
            
        # 构建原始查询的 TF-IDF 权重向量
        original_query_weights={}
        for w, tf in query_tf.items():
            original_query_weights[w]=tf*idf[w]
            
        # 第一次检索
        initial_ranking=execute_search(original_query_weights,inverted_index,image_norms)
        
        # 如果什么都没搜到，直接跳过
        if not initial_ranking:
            system_results[query_name]=[]
            continue
            
        # AQE 核心逻辑
        # 提取第一次检索排名前K的图片名（即盲目信任它们是对的）
        top_k_imgs=[img for img,score in initial_ranking[:top_k_expand]]
        
        # 复制一份原始查询向量，准备进行特征融合
        expanded_weights=original_query_weights.copy()
        
        # 遍历倒排表，把这K张图的特征全部“吸”过来，取平均后加到查询向量上
        for w,db_docs in inverted_index.items():
            sum_expanded_weight=0.0
            for img in top_k_imgs:
                if img in db_docs:
                    sum_expanded_weight+=db_docs[img]
            
            if sum_expanded_weight>0:
                # 公式：Q_new=Q_old+(Sum(D_1...D_k)/K)
                expanded_weights[w]=expanded_weights.get(w,0)+(sum_expanded_weight/top_k_expand)
                
        # 第二次检索
        final_ranking=execute_search(expanded_weights,inverted_index,image_norms)
        
        # 保存重排后的纯图片名列表
        system_results[query_name]=[f"{img}.jpg" for img,score in final_ranking[:100]]
        
    print("\n正在计算mAP分数……")
    mAP_score=evaluate_system(PARSED_GT_PATH,system_results)
    
    print(f"AQE扩展系统构建完毕，最终mAP得分:{mAP_score:.4f}")

# if __name__ == '__main__':
#     aqe_retrieval(top_k_expand=5)