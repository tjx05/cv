import os

# ==========================================
# 环境变量设置 (必须在导入 numpy 前执行)
# ==========================================
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "24"  # ✅ 修复OpenBLAS警告，从32改为24
os.environ["MKL_NUM_THREADS"] = "32"

import glob
import numpy as np
from tqdm import tqdm
import sys

# 将工程根目录加入系统路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import FEATURES_DIR, VOCAB_PATH, VOCAB_SIZE_K, KMEANS_SAMPLE_RATE, KMEANS_BATCH_SIZE
from model.minibatchKMeans import CustomMiniBatchKMeans

def load_and_sample_features(features_dir, sample_rate):
    """
    负责从磁盘读取特征文件并进行降采样
    """
    npy_files = glob.glob(os.path.join(features_dir, '*.npy'))
    print(f"找到 {len(npy_files)} 个特征文件")

    sampled_features = []
    for npy_file in tqdm(npy_files, desc="加载特征"):
        try:
            descs = np.load(npy_file)
            if len(descs) == 0:
                continue
            # 计算采样数量
            n = max(1, int(len(descs) * sample_rate))
            idx = np.random.choice(len(descs), n, replace=False)
            sampled_features.append(descs[idx])
        except Exception as e:
            print(f"跳过 {npy_file}: {e}")

    # 将所有采样到的特征拼接成一个巨大的特征矩阵
    all_features = np.vstack(sampled_features).astype(np.float32)
    print(f"特征总数: {len(all_features)}，内存: {all_features.nbytes/1024**3:.2f} GB")
    return all_features

def main():
    print(f"K={VOCAB_SIZE_K}, 采样率={KMEANS_SAMPLE_RATE}, batch={KMEANS_BATCH_SIZE}")

    # 1. 业务逻辑：加载与清洗数据
    all_features = load_and_sample_features(FEATURES_DIR, KMEANS_SAMPLE_RATE)

    # 2. 算法调用：初始化 KMeans 模型
    print(f"\n🚀 开始初始化聚类引擎...")
    vocab_model = CustomMiniBatchKMeans(
        vocab_size=VOCAB_SIZE_K,
        batch_size=KMEANS_BATCH_SIZE,
        n_epochs=3  # 过3遍数据，质量接近全局 fit()
    )

    # 3. 算法调用：执行训练
    vocab_model.train(all_features)

    # 4. 数据持久化：保存结果
    vocab_model.save(VOCAB_PATH)
    print(f"\n🎉 词典已成功保存至: {VOCAB_PATH}")

if __name__ == '__main__':
    main()