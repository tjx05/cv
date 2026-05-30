# -*- coding: utf-8 -*-
"""
模块名称: AKM (Approximate K-Means)
复现论文: "Object retrieval with large vocabularies and fast spatial matching"
         Philbin et al., CVPR 2007

论文核心设计:
  1. 8 棵随机化 k-d 树森林，每轮迭代重建
  2. 共享优先队列搜索，记录到判别边界的距离
  3. 固定探索节点数截断（max_checks）
  4. 向量化 M-step + 空簇随机重分配
"""

import os
import numpy as np
import heapq
from tqdm import tqdm


# ============================================================
#  随机化 k-d 树
# ============================================================

class _KDNode:
    __slots__ = ['split_dim', 'split_val', 'left', 'right', 'indices']

    def __init__(self):
        self.split_dim = None
        self.split_val = None
        self.left      = None
        self.right     = None
        self.indices   = None  # 叶节点存储点索引


class _RandomizedKDTree:
    """
    单棵随机化 k-d 树
    - 分裂维：从方差最高的 top_rand_dims 个维度中随机选一个（论文原文）
    - 分裂值：中位数附近随机选取一个点的值（论文原文）
    """

    def __init__(self, top_rand_dims=5):
        self.top_rand_dims = top_rand_dims
        self.root   = None
        self.points = None

    def build(self, points):
        """points: (K, D) float32，当前聚类中心"""
        self.points = points
        self.root   = self._build(np.arange(len(points)))

    def _build(self, indices):
        node = _KDNode()

        if len(indices) <= 1:
            node.indices = indices
            return node

        subset    = self.points[indices]
        variances = np.var(subset, axis=0)

        # 从方差最高的 top_rand_dims 个维度中随机选一维
        k         = min(self.top_rand_dims, len(variances))
        top_dims  = np.argpartition(variances, -k)[-k:]
        split_dim = int(np.random.choice(top_dims))

        # 分裂值：中位数附近随机选一个点的值
        vals      = subset[:, split_dim]
        sorted_v  = np.sort(vals)
        mid       = len(sorted_v) // 2
        lo        = max(0,                 mid - max(1, len(sorted_v) // 10))
        hi        = min(len(sorted_v) - 1, mid + max(1, len(sorted_v) // 10))
        split_val = float(sorted_v[np.random.randint(lo, hi + 1)])

        node.split_dim = split_dim
        node.split_val = split_val

        left_mask  = vals <= split_val
        right_mask = ~left_mask

        # 防止退化（所有点落在同一侧）
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            node.indices = indices
            return node

        node.left  = self._build(indices[left_mask])
        node.right = self._build(indices[right_mask])
        return node


# ============================================================
#  随机化 k-d 树森林
# ============================================================

class _RandomizedKDForest:
    """
    8 棵随机化 k-d 树森林 + 共享优先队列搜索
    完整复现论文 Section 3.1 的近似最近邻流程
    """

    def __init__(self, n_trees=8, top_rand_dims=5, max_checks=200):
        self.n_trees       = n_trees
        self.top_rand_dims = top_rand_dims
        self.max_checks    = max_checks
        self.trees         = []
        self.points        = None

    def build(self, points):
        """每轮迭代开始时重建（论文原文要求）"""
        self.points = points.astype(np.float32)
        self.trees  = []
        for _ in range(self.n_trees):
            t = _RandomizedKDTree(top_rand_dims=self.top_rand_dims)
            t.build(self.points)
            self.trees.append(t)

    def query_batch(self, queries):
        """
        批量近似最近邻查询
        queries: (N, D) float32
        返回:    labels (N,) int64
        """
        N      = len(queries)
        labels = np.zeros(N, dtype=np.int64)

        for qi in range(N):
            q          = queries[qi]
            best_dist  = np.inf
            best_label = 0
            checks     = 0
            visited    = set()

            # 所有树的根节点压入共享优先队列（论文原文）
            pq = []
            for tree in self.trees:
                heapq.heappush(pq, (0.0, id(tree.root), tree.root))

            while pq and checks < self.max_checks:
                cost, _, node = heapq.heappop(pq)

                # 早剪枝
                if cost >= best_dist:
                    break

                # 叶节点：计算精确 L2 距离
                if node.left is None and node.right is None:
                    if node.indices is not None:
                        for idx in node.indices:
                            if idx in visited:
                                continue
                            visited.add(idx)
                            checks += 1
                            d = float(np.sum((q - self.points[idx]) ** 2))
                            if d < best_dist:
                                best_dist  = d
                                best_label = int(idx)
                    continue

                # 内部节点：主方向直接压入，备选方向记录到边界距离²
                diff = float(q[node.split_dim]) - node.split_val
                if diff <= 0:
                    primary, secondary = node.left, node.right
                else:
                    primary, secondary = node.right, node.left

                if primary is not None:
                    heapq.heappush(pq, (cost, id(primary), primary))
                if secondary is not None:
                    heapq.heappush(pq, (cost + diff ** 2, id(secondary), secondary))

            labels[qi] = best_label

        return labels


# ============================================================
#  AKM 主类（纯 Python 版，忠实复现论文）
# ============================================================

class ApproximateKMeans:
    """
    论文忠实复现版 AKM，仅依赖 numpy
    """

    def __init__(self, n_clusters, max_iter=100, tol=1.0,
                 n_trees=8, max_checks=200, top_rand_dims=5,
                 batch_size=50000):
        """
        n_clusters:    视觉词汇表大小 K
        max_iter:      最大迭代次数
        tol:           收敛阈值（聚类中心总漂移量）
        n_trees:       随机 k-d 树棵数，论文指定 8
        max_checks:    每次查询最多探索节点数
        top_rand_dims: 随机分裂维候选数
        batch_size:    E-step 分批大小
        """
        self.n_clusters    = n_clusters
        self.max_iter      = max_iter
        self.tol           = tol
        self.n_trees       = n_trees
        self.max_checks    = max_checks
        self.top_rand_dims = top_rand_dims
        self.batch_size    = batch_size
        self.cluster_centers_ = None

    def fit(self, X):
        """
        训练视觉词汇表
        X: (N, D) float32
        """
        X    = X.astype(np.float32)
        N, D = X.shape

        if N < self.n_clusters:
            raise ValueError(
                f"样本数 N={N} 小于聚类数 K={self.n_clusters}，"
                f"请减小 VOCAB_SIZE 或增加特征数量"
            )

        # 随机采样初始化（与论文一致）
        np.random.seed(42)
        init_idx = np.random.choice(N, self.n_clusters, replace=False)
        self.cluster_centers_ = X[init_idx].copy()

        forest = _RandomizedKDForest(
            n_trees       = self.n_trees,
            top_rand_dims = self.top_rand_dims,
            max_checks    = self.max_checks
        )

        for iteration in range(self.max_iter):
            print(f"─── 迭代 {iteration+1:3d}/{self.max_iter} ───")

            # 每轮在当前聚类中心上重建树森林（论文原文）
            print(f"  构建 {self.n_trees} 棵随机 k-d 树...")
            forest.build(self.cluster_centers_)

            # E-step：近似最近邻分配（分批）
            labels    = np.zeros(N, dtype=np.int64)
            n_batches = (N + self.batch_size - 1) // self.batch_size

            for b in tqdm(range(n_batches), desc="  E-step", leave=False):
                s = b * self.batch_size
                e = min(s + self.batch_size, N)
                labels[s:e] = forest.query_batch(X[s:e])

            # M-step：向量化重计算聚类中心
            new_centers = np.zeros((self.n_clusters, D), dtype=np.float32)
            np.add.at(new_centers, labels, X)
            counts      = np.bincount(labels, minlength=self.n_clusters)
            safe_counts = np.maximum(counts, 1).reshape(-1, 1).astype(np.float32)
            new_centers /= safe_counts

            # 空簇：随机从数据中补充
            empty = np.where(counts == 0)[0]
            if len(empty) > 0:
                rep = np.random.choice(N, len(empty), replace=False)
                new_centers[empty] = X[rep]
                print(f"  ⚠️  {len(empty)} 个空簇已重分配")

            # 收敛检测
            shift = float(np.sum(np.linalg.norm(
                self.cluster_centers_ - new_centers, axis=1
            )))
            self.cluster_centers_ = new_centers
            print(f"  中心漂移: {shift:.4f}  空簇: {len(empty)}")

            if shift < self.tol:
                print(f"\n🎉 第 {iteration+1} 轮收敛（漂移={shift:.4f} < tol={self.tol}）")
                break

        print(f"\n✅ AKM 训练完成，词汇表大小: {self.n_clusters}")
        return self

    def predict(self, X):
        """
        将特征量化为视觉单词 ID
        X: (N, D) float32
        返回: labels (N,) int64
        """
        X = X.astype(np.float32)
        N = len(X)

        forest = _RandomizedKDForest(
            n_trees       = self.n_trees,
            top_rand_dims = self.top_rand_dims,
            max_checks    = self.max_checks
        )
        forest.build(self.cluster_centers_)

        labels    = np.zeros(N, dtype=np.int64)
        n_batches = (N + self.batch_size - 1) // self.batch_size

        for b in tqdm(range(n_batches), desc="量化特征"):
            s = b * self.batch_size
            e = min(s + self.batch_size, N)
            labels[s:e] = forest.query_batch(X[s:e])

        return labels

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.cluster_centers_)
        print(f"💾 词汇表已保存: {path}  shape={self.cluster_centers_.shape}")

    @classmethod
    def load(cls, path, **kwargs):
        centers = np.load(path)
        model   = cls(n_clusters=len(centers), **kwargs)
        model.cluster_centers_ = centers.astype(np.float32)
        print(f"📂 词汇表已加载: {path}  shape={centers.shape}")
        return model


# ============================================================
#  Faiss 加速版（有 GPU 时推荐）
# ============================================================

class ApproximateKMeansFaiss:
    """
    Faiss GPU 加速版 AKM
    安装: pip install faiss-gpu
    A40 上 K=10000、120万特征约 3~5 分钟完成
    """

    def __init__(self, n_clusters, max_iter=100, tol=1.0,
                 use_gpu=True, ef_search=64, ef_construction=200):
        self.n_clusters      = n_clusters
        self.max_iter        = max_iter
        self.tol             = tol
        self.use_gpu         = use_gpu
        self.ef_search       = ef_search
        self.ef_construction = ef_construction
        self.cluster_centers_ = None

    def _build_index(self, centers):
        import faiss
        D     = centers.shape[1]
        index = faiss.IndexHNSWFlat(D, 32)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch       = self.ef_search
        if self.use_gpu:
            try:
                res   = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                print("  ✅ 使用 GPU 加速")
            except Exception:
                print("  ⚠️  GPU 不可用，回退到 CPU")
        index.add(centers)
        return index

    def fit(self, X):
        import faiss
        X    = X.astype(np.float32)
        N, D = X.shape

        if N < self.n_clusters:
            raise ValueError(
                f"样本数 N={N} 小于聚类数 K={self.n_clusters}，"
                f"请减小 VOCAB_SIZE 或增加特征数量"
            )

        np.random.seed(42)
        idx = np.random.choice(N, self.n_clusters, replace=False)
        self.cluster_centers_ = X[idx].copy()

        for iteration in range(self.max_iter):
            print(f"─── 迭代 {iteration+1:3d}/{self.max_iter} ───")

            index = self._build_index(self.cluster_centers_)
            distances, labels = index.search(X, 1)
            labels = labels.ravel()

            new_centers = np.zeros((self.n_clusters, D), dtype=np.float32)
            np.add.at(new_centers, labels, X)
            counts      = np.bincount(labels, minlength=self.n_clusters)
            safe_counts = np.maximum(counts, 1).reshape(-1, 1).astype(np.float32)
            new_centers /= safe_counts

            empty = np.where(counts == 0)[0]
            if len(empty) > 0:
                rep = np.random.choice(N, len(empty), replace=False)
                new_centers[empty] = X[rep]
                print(f"  ⚠️  {len(empty)} 个空簇已重分配")

            shift = float(np.sum(np.linalg.norm(
                self.cluster_centers_ - new_centers, axis=1
            )))
            inertia = float(np.sum(distances))
            self.cluster_centers_ = new_centers
            print(f"  中心漂移: {shift:.4f}  Inertia: {inertia:.2f}  空簇: {len(empty)}")

            if shift < self.tol:
                print(f"\n🎉 第 {iteration+1} 轮收敛")
                break

        print(f"\n✅ AKM-Faiss 训练完成，词汇表大小: {self.n_clusters}")
        return self

    def predict(self, X):
        import faiss
        X     = X.astype(np.float32)
        D     = self.cluster_centers_.shape[1]
        index = faiss.IndexFlatL2(D)
        index.add(self.cluster_centers_)
        _, labels = index.search(X, 1)
        return labels.ravel().astype(np.int64)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.cluster_centers_)
        print(f"💾 词汇表已保存: {path}  shape={self.cluster_centers_.shape}")

    @classmethod
    def load(cls, path, **kwargs):
        centers = np.load(path)
        model   = cls(n_clusters=len(centers), **kwargs)
        model.cluster_centers_ = centers.astype(np.float32)
        print(f"📂 词汇表已加载: {path}  shape={centers.shape}")
        return model