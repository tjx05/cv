import os
import json
import pickle
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from collections import defaultdict

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,BASE_DIR)

from config import IMAGE_DIR,VOCAB_PATH,INDEX_PATH,PARSED_GT_PATH,SIFT_MAX_FEATURES,USE_ROOT_SIFT
from model.ROOTSIFT_feature import RootSIFTExtractor

def visualize_query_results(query_key='all_souls_1'):
    """
    针对单个 Query 进行检索，并可视化 Top-5 结果
    """
    print(f"正在加载系统数据，准备可视化: {query_key} ...")
    with open(VOCAB_PATH,'rb') as f:
        kmeans=pickle.load(f)
    with open(INDEX_PATH,'rb') as f: 
        index_data=pickle.load(f)
    with open(PARSED_GT_PATH,'r',encoding='utf-8') as f: 
        gt_data=json.load(f)
    
    if query_key not in gt_data:
        print(f"找不到查询任务 {query_key}，请检查名字！")
        return

    gt_info=gt_data[query_key]
    query_img_name=gt_info['query_img']
    bbox=gt_info['bbox']
    good_imgs=set(gt_info['good']+gt_info['ok']) # 正确的图片集合
    
    extractor=RootSIFTExtractor(max_features=SIFT_MAX_FEATURES,use_rootsift=USE_ROOT_SIFT)
    inverted_index=index_data['inverted_index']
    idf=index_data['idf']
    image_norms=index_data['image_norms']
    
    query_img_path=os.path.join(IMAGE_DIR,query_img_name)
    kps,descs=extractor.extract(query_img_path, bbox=bbox)
    
    words=kmeans.predict(descs)
    query_tf=defaultdict(int)
    for w in words: 
        query_tf[w] += 1
        
    query_norm_sq=0.0
    query_weights={}
    for w,tf in query_tf.items():
        weight=tf*idf[w]
        query_weights[w]=weight
        query_norm_sq+=weight**2
    query_norm=math.sqrt(query_norm_sq)
    
    scores=defaultdict(float)
    for w,q_weight in query_weights.items():
        if w in inverted_index:
            for db_img, db_weight in inverted_index[w].items():
                scores[db_img]+=q_weight*db_weight
                
    final_scores={img: dot/(query_norm*image_norms.get(img, 1.0)) for img,dot in scores.items()}
    ranked_list=sorted(final_scores.items(),key=lambda x:x[1],reverse=True)[:5] # 只取前5名
    
    # 画图
    fig,axes=plt.subplots(1,6,figsize=(20, 4))
    fig.suptitle(f"Image Retrieval Results for: {query_key}", fontsize=16, fontweight='bold')
    
    # 画 Query 原图 (加黄色 Bounding Box)
    q_img = cv2.imread(query_img_path)
    q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(q_img)
    axes[0].set_title("Query Image", color='blue')
    axes[0].axis('off')
    
    # 画框
    xmin, ymin, xmax, ymax = bbox
    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=3, edgecolor='yellow', facecolor='none')
    axes[0].add_patch(rect)
    
    # 画 Top-5 检索结果
    for i, (db_img_name, score) in enumerate(ranked_list):
        ax = axes[i + 1]
        img_path = os.path.join(IMAGE_DIR, f"{db_img_name}.jpg")
        
        if os.path.exists(img_path):
            res_img = cv2.imread(img_path)
            res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
            ax.imshow(res_img)
        
        # 判断是对是错
        full_img_name = f"{db_img_name}.jpg"
        if full_img_name in good_imgs:
            ax.set_title(f"Rank {i+1}\n(Correct)", color='green', fontweight='bold')
            # 绿框
            for spine in ax.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(5)
        else:
            ax.set_title(f"Rank {i+1}\n(Wrong)", color='red', fontweight='bold')
            # 红框
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(5)
                
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    # 保存图片到 outputs 目录
    out_img_path = os.path.join(BASE_DIR, 'outputs', f'visualize_{query_key}.png')
    plt.savefig(out_img_path, dpi=300)
    print(f"可视化结果已保存至: {out_img_path}")
    plt.show()

if __name__=='__main__':
    visualize_query_results(query_key='all_souls_1')