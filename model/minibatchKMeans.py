import os
import pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

class CustomMiniBatchKMeans:
    """
    封装的 MiniBatchKMeans 视觉词典训练器
    """
    def __init__(self, vocab_size, batch_size, n_epochs=3, random_state=42):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # 初始化底层的 sklearn 模型
        self.kmeans = MiniBatchKMeans(
            n_clusters=vocab_size,
            batch_size=batch_size,
            init='random',
            n_init=1,
            max_iter=100,
            max_no_improvement=10,
            tol=0.0,
            verbose=0,
            random_state=random_state,
            compute_labels=False,
            reassignment_ratio=0.01
        )

    def train(self, all_features):
        """
        执行模型训练 (支持 partial_fit 的多 Epoch 迭代)
        """
        n_samples = len(all_features)
        n_batches = int(np.ceil(n_samples / self.batch_size))

        with tqdm(total=n_batches * self.n_epochs, desc=f"🔧 KMeans聚类 K={self.vocab_size}", unit="batch") as pbar:
            for epoch in range(self.n_epochs):
                # 每个 epoch 打乱一次数据顺序，提升收敛质量
                shuffle_idx = np.random.permutation(n_samples)
                shuffled = all_features[shuffle_idx]

                for i in range(n_batches):
                    start = i * self.batch_size
                    end = min(start + self.batch_size, n_samples)
                    batch = shuffled[start:end]
                    
                    # 核心：局部拟合
                    self.kmeans.partial_fit(batch)

                    if hasattr(self.kmeans, 'inertia_'):
                        pbar.set_postfix({
                            'epoch': f'{epoch+1}/{self.n_epochs}',
                            'inertia': f'{self.kmeans.inertia_:.4f}'
                        })
                    pbar.update(1)

    def save(self, vocab_path):
        """
        序列化保存训练好的模型
        """
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, 'wb') as f:
            pickle.dump(self.kmeans, f)