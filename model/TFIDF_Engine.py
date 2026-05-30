# -*- coding: utf-8 -*-
"""
模块名称: TFIDF_Engine (TF-IDF 倒排检索引擎)
实现特性: 纯手工构建，包含高维稀疏特征的量化、离线倒排建库、向量模长预计算以及在线极速检索。
"""
import os
import glob
import numpy as np
import pickle
import math
from collections import defaultdict
from tqdm import tqdm

def build_inverted_index(features_dir, vocab_path, index_path):
    """离线阶段：构建倒排索引并序列化至硬盘"""
    if not os.path.exists(vocab_path):
        print(f"❌ 找不到词典模型: {vocab_path}")
        return

    print("📖 加载视觉词典模型...")
    with open(vocab_path, 'rb') as f:
        kmeans = pickle.load(f)
    
    vocab_size = kmeans.n_clusters
    print(f"当前词汇量 K = {vocab_size}")

    npy_files = glob.glob(os.path.join(features_dir, '*.npy'))
    total_images = len(npy_files)
    
    # 初始化核心数据结构
    inverted_index = defaultdict(dict)
    df = np.zeros(vocab_size)
    image_word_counts = {}
    
    print("\n[第一阶段] 将所有图像的局部特征量化为离散的视觉单词 (Word Quantization)...")
    for npy_file in tqdm(npy_files, desc="量化特征"):
        base_name = os.path.splitext(os.path.basename(npy_file))[0]
        descs = np.load(npy_file)
        
        if descs is None or len(descs) == 0:
            continue
            
        # 用 KMeans 预测视觉单词 ID
        words = kmeans.predict(descs)
        
        # 统计单张图词频 (TF)
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
            
        image_word_counts[base_name] = word_counts
        
        # 统计文档频率 (DF)
        for w in word_counts.keys():
            df[w] += 1

    print("\n[第二阶段] 计算 TF-IDF 权重并建立倒排链表...")
    # 计算全局 IDF
    idf = np.log(total_images / (df + 1e-7))
    image_norms = {} 
    
    for img_name, word_counts in tqdm(image_word_counts.items(), desc="计算权重"):
        img_norm_sq = 0.0
        for word_id, tf in word_counts.items():
            weight = tf * idf[word_id]
            # 核心：挂载倒排链表
            inverted_index[word_id][img_name] = weight
            img_norm_sq += weight ** 2
            
        # 离线预计算向量 L2 长度，加速查询
        image_norms[img_name] = math.sqrt(img_norm_sq)
        
    print("\n[第三阶段] 保存倒排索引结构至硬盘...")
    index_data = {
        'inverted_index': dict(inverted_index), 
        'idf': idf,
        'image_norms': image_norms,
        'vocab_size': vocab_size
    }
    
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, 'wb') as f:
        pickle.dump(index_data, f)
        
    print(f"🎉 倒排索引构建成功！已保存至 {index_path}")


def compute_query_weights(descs, kmeans, idf):
    """在线阶段：将查询图像的连续特征，转化为离散的 TF-IDF 权重字典"""
    words = kmeans.predict(descs)
    query_tf = defaultdict(int)
    for w in words: 
        query_tf[w] += 1
        
    query_weights = {w: tf * idf[w] for w, tf in query_tf.items()}
    return query_weights


def execute_bow_search(query_weights, inverted_index, image_norms, top_n):
    """在线阶段：基于 TF-IDF 与余弦相似度的稀疏倒排极速查询"""
    scores = defaultdict(float)
    query_norm_sq = sum(w ** 2 for w in query_weights.values())
    query_norm = math.sqrt(query_norm_sq)
    if query_norm == 0: return []

    for w, q_weight in query_weights.items():
        if w in inverted_index:
            for db_img, db_weight in inverted_index[w].items():
                scores[db_img] += q_weight * db_weight
                
    # 归一化点积，得到余弦相似度
    final_scores = {img: dot / (query_norm * image_norms.get(img, 1.0)) for img, dot in scores.items()}
    
    # 返回按分数降序排列的 top_n 列表
    return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]