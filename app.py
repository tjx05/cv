from flask import Flask,render_template,request,jsonify,send_from_directory
import os
import base64
import cv2
import numpy as np
import tempfile

from config import IMAGE_DIR,KEYPOINTS_DIR
from scripts.ransac_aqe_retrieval import search_single_query

app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=16*1024*1024  # 限制16MB

# 临时保存上传图片的目录
UPLOAD_FOLDER='uploads'
os.makedirs(UPLOAD_FOLDER,exist_ok=True)

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/search',methods=['POST'])
def search():
    """图像检索接口"""
    # 获取上传的图片
    if 'image' not in request.files:
        return jsonify({'success': False,'error': '未收到图片'})
    
    file = request.files['image']
    if file.filename=='':
        return jsonify({'success': False,'error': '未选择文件'})
    
    # 获取返回数量参数
    top_k=int(request.form.get('top_k',10))
    threshold=float(request.form.get('threshold',5.0)) 

    # 保存临时文件
    suffix = os.path.splitext(file.filename)[1] or '.jpg'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path=tmp.name

    try:
        # 单张检索函数
        results=search_single_query(tmp_path,bbox=None,final_top_n=top_k,ransac_thresh=threshold)
        
        # 格式化返回结果
        formatted_results=[]
        for img_name,inliers,bow_score,match_pts in results:
            formatted_results.append({
                'filename': f"{img_name}.jpg",
                'score': round(bow_score, 4),
                'inliers': inliers,
                'image_url': f'/image/{img_name}.jpg',
                'keypoints_url': f'/keypoints/{img_name}.npy',
                'matches': match_pts
            })
        
        return jsonify({'success': True,'results': formatted_results})
    
    except Exception as e:
        return jsonify({'success': False,'error': str(e)})
    
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
@app.route('/image/<filename>.jpg')
def serve_image(filename):
    """提供数据库图片访问"""
    return send_from_directory(IMAGE_DIR,f'{filename}.jpg')

@app.route('/keypoints/<filename>.npy')
def serve_keypoints(filename):
    """提供关键点文件访问"""
    return send_from_directory(KEYPOINTS_DIR, f'{filename}.npy')



if __name__=='__main__':
    app.run(debug=True,host='127.0.0.1',port=5000)