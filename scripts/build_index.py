import os
import glob
import numpy as np
import pickle
import math
from collections import defaultdict
from tqdm import tqdm
import sys

# 导入全局配置
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,BASE_DIR)
from config import FEATURES_DIR,VOCAB_PATH,INDEX_PATH

def build_inverted_index(features_dir,vocab_path,index_path):
    if not os.path.exists(vocab_path):
        print(f"❌ 找不到词典模型: {vocab_path}")
        return

    print("加载视觉词典模型...")
    with open(vocab_path,'rb') as f:
        kmeans=pickle.load(f)
    
    vocab_size=kmeans.n_clusters
    print(f"当前词汇量 K={vocab_size}")

    npy_files=glob.glob(os.path.join(features_dir, '*.npy'))
    total_images=len(npy_files)
    
    # 初始化核心数据结构
    # inverted_index: 词汇ID -> {图片名:权重}
    inverted_index=defaultdict(dict)
    
    # df: 记录每个词出现在多少张不同的图片中
    df=np.zeros(vocab_size)
    
    # 临时存储每张图的词频
    image_word_counts={}
    
    print("\n[第一阶段] 将所有图像的局部特征量化为离散的视觉单词...")
    for npy_file in tqdm(npy_files,desc="量化特征"):
        base_name=os.path.splitext(os.path.basename(npy_file))[0]
        descs=np.load(npy_file)
        
        if descs is None or len(descs)==0:
            continue
            
        # 用KMeans预测，把128维特征变成1维单词ID
        words=kmeans.predict(descs)
        
        # 统计这张图片内每个单词出现的次数
        word_counts={}
        for w in words:
            word_counts[w]=word_counts.get(w,0)+1
            
        image_word_counts[base_name]=word_counts
        
        # 统计包含该单词的文档数量
        for w in word_counts.keys():
            df[w]+=1

    print("\n[第二阶段] 计算TF-IDF权重并建立倒排链表...")
    # 计算全局 IDF
    # 极少出现的词（区分度高）赋予极大权重，泛滥的词赋予极小权重
    idf=np.log(total_images/(df+1e-7))
    
    image_norms={} # 用于存储每张图的L2范数，查询时做余弦相似度归一化
    
    for img_name,word_counts in tqdm(image_word_counts.items(),desc="计算权重"):
        img_norm_sq=0.0
        
        for word_id,tf in word_counts.items():
            # 计算TF-IDF
            weight=tf*idf[word_id]
            
            # 将图片及其权重挂载到该词汇的倒排链表下
            inverted_index[word_id][img_name]=weight
            
            # 累加平方和
            img_norm_sq+=weight**2
            
        # 计算该图片向量的L2长度
        image_norms[img_name]=math.sqrt(img_norm_sq)
        
    print("\n[第三阶段] 保存倒排索引结构至硬盘...")
    index_data={
        'inverted_index': dict(inverted_index), # 转为普通字典加速读取
        'idf': idf,
        'image_norms': image_norms,
        'vocab_size': vocab_size
    }
    
    os.makedirs(os.path.dirname(index_path),exist_ok=True)
    with open(index_path,'wb') as f:
        pickle.dump(index_data,f)
        
    print(f"倒排索引构建成功！已保存至 {index_path}")
    print(f"总计包含 {len(index_data['inverted_index'])} 个有效检索链表。")

if __name__ == '__main__':
    build_inverted_index(FEATURES_DIR, VOCAB_PATH, INDEX_PATH)