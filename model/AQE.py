# -*- coding: utf-8 -*-
"""
模块名称: AQE (Average Query Expansion) 平均查询扩展模块
实现特性: 纯手工编写，无第三方库依赖，基于稀疏字典的自适应多视图特征聚合
"""
import os
import sys

# 动态获取项目根目录并加入系统路径，确保能正确导入 config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import TOP_K_EXPAND  

def average_query_expansion(original_query_weights, top_k_imgs_verified, inverted_index, top_k_expand=TOP_K_EXPAND):
    """
    纯手写的平均查询扩展算法核心逻辑
    
    参数:
    ----------
    original_query_weights : dict
        原始查询图像的 TF-IDF 权重字典 {word_id: weight}
    top_k_imgs_verified : list
        经过第一轮手写 RANSAC 严苛空间校验的高置信度正样本图像名称列表
    inverted_index : dict
        全局倒排索引大字典 {word_id: {img_name: weight}}
    top_k_expand : int, 可选
        用于参与特征扩展的图像数量 (默认 5)
        
    返回:
    ----------
    expanded_weights : dict
        融合多视图特征后膨胀得到的“超级查询向量”权重字典
    """
    # 1. 以原始查询特征作为融合基底，确保核心特征不丢失
    expanded_weights = original_query_weights.copy()
    
    # 实际传入的高置信度图像数量
    actual_expand_num = len(top_k_imgs_verified)
    
    # 如果没有找到高置信度的空间线人，直接返回原始特征避免遭受噪声污染
    if actual_expand_num == 0 or top_k_expand == 0:
        return expanded_weights
        
    # 2. 遍历全局倒排索引，动态反查并聚合这 K 张图像在离散空间中的特征权重
    for word_id, db_docs in inverted_index.items():
        sum_expanded_weight = 0.0
        
        # 只检索当前被锁定的高置信度图像
        for img in top_k_imgs_verified:
            if img in db_docs: 
                sum_expanded_weight += db_docs[img]
                
        # 3. 执行均值池化 (Average Pooling)
        if sum_expanded_weight > 0:
            # 除以实际参与融合的图像数量，避免权重被空置位稀释
            avg_increment = sum_expanded_weight / actual_expand_num
            expanded_weights[word_id] = expanded_weights.get(word_id, 0) + avg_increment
            
    return expanded_weights

