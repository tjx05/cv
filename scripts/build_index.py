import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0, BASE_DIR)

from config import FEATURES_DIR, VOCAB_PATH, INDEX_PATH
from model.TFIDF_Engine import build_inverted_index

if __name__ == '__main__':
    print(">>> 启动离线建库流水线...")
    build_inverted_index(FEATURES_DIR, VOCAB_PATH, INDEX_PATH)
    print(">>> 建库流水线执行完毕！")