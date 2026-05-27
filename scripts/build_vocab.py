import os
os.environ["OMP_NUM_THREADS"]="32"
os.environ["OPENBLAS_NUM_THREADS"]="24"  # 修复OpenBLAS警告，从32改为24
os.environ["MKL_NUM_THREADS"]="32"

import glob
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import sys

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,BASE_DIR)
from config import FEATURES_DIR,VOCAB_PATH,VOCAB_SIZE_K,KMEANS_SAMPLE_RATE,KMEANS_BATCH_SIZE


def train_visual_vocabulary(features_dir,vocab_path,vocab_size,sample_rate):
    npy_files=glob.glob(os.path.join(features_dir,'*.npy'))
    print(f"找到{len(npy_files)}个特征文件")

    # 加载特征
    sampled_features=[]
    for npy_file in tqdm(npy_files,desc="加载特征"):
        try:
            descs=np.load(npy_file)
            if len(descs)==0:
                continue
            n=max(1,int(len(descs)*sample_rate))
            idx=np.random.choice(len(descs),n,replace=False)
            sampled_features.append(descs[idx])
        except Exception as e:
            print(f"跳过 {npy_file}: {e}")

    all_features=np.vstack(sampled_features).astype(np.float32)
    print(f"特征总数: {len(all_features)}，内存: {all_features.nbytes/1024**3:.2f} GB")

    # 聚类
    print(f"\n开始聚类K={vocab_size}...")
    kmeans=MiniBatchKMeans(
        n_clusters=vocab_size,
        batch_size=KMEANS_BATCH_SIZE,
        init='random',
        n_init=1,
        max_iter=100,
        max_no_improvement=10,
        tol=0.0,
        verbose=0,
        random_state=42,
        compute_labels=False,
        reassignment_ratio=0.01
    )

    n_samples=len(all_features)
    n_batches=int(np.ceil(n_samples/KMEANS_BATCH_SIZE))
    N_EPOCHS=3  # 过3遍数据，质量接近fit()

    with tqdm(total=n_batches*N_EPOCHS,desc=f"KMeans聚类 K={vocab_size}",unit="batch") as pbar:
        for epoch in range(N_EPOCHS):
            shuffle_idx=np.random.permutation(n_samples)
            shuffled=all_features[shuffle_idx]

            for i in range(n_batches):
                start=i*KMEANS_BATCH_SIZE
                end=min(start+KMEANS_BATCH_SIZE,n_samples)
                batch=shuffled[start:end]
                kmeans.partial_fit(batch)

                if hasattr(kmeans,'inertia_'):
                    pbar.set_postfix({
                        'epoch': f'{epoch+1}/{N_EPOCHS}',
                        'inertia': f'{kmeans.inertia_:.4f}'
                    })
                pbar.update(1)

    # 保存
    os.makedirs(os.path.dirname(vocab_path),exist_ok=True)
    with open(vocab_path,'wb') as f:
        pickle.dump(kmeans,f)
    print(f"\n词典已保存至{vocab_path}")


if __name__=='__main__':
    print(f"K={VOCAB_SIZE_K},采样率={KMEANS_SAMPLE_RATE},batch={KMEANS_BATCH_SIZE}")
    train_visual_vocabulary(
        features_dir=FEATURES_DIR,
        vocab_path=VOCAB_PATH,
        vocab_size=VOCAB_SIZE_K,
        sample_rate=KMEANS_SAMPLE_RATE
    )