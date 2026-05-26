// 滑块实时更新
const sliderN = document.getElementById('sliderN');
const valN = document.getElementById('valN');
const sliderTh = document.getElementById('sliderTh');
const valTh = document.getElementById('valTh');

if (sliderN) {
    sliderN.oninput = function() {
        valN.innerText = this.value;
    }
}

if (sliderTh) {
    sliderTh.oninput = function() {
        valTh.innerText = this.value;
    }
}

// 上传与检索功能
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('file-input');
const preImg = document.getElementById('pre-img');
const emptyTip = document.getElementById('emptyTip');
const searchBtn = document.getElementById('search');
const resultBox = document.getElementById('resultBox');

if (uploadBtn) {
    uploadBtn.addEventListener('click', () => fileInput.click());
}

if (fileInput) {
    fileInput.addEventListener('change', function(e) {
        let file = e.target.files[0];
        if (!file) return;
        let url = URL.createObjectURL(file);
        preImg.src = url;
        preImg.style.display = 'block';
        if (emptyTip) emptyTip.style.display = 'none';
    });
}

if (searchBtn) {
    searchBtn.addEventListener('click', async () => {
        if (!preImg.src || preImg.style.display === 'none') {
            alert('请先上传图片');
            return;
        }

        // 获取参数
        const topK = sliderN ? parseInt(sliderN.value) : 4;
        const threshold = sliderTh ? parseInt(sliderTh.value) : 50;

        // 显示加载状态
        resultBox.innerHTML = `<div class="empty-result">检索中，请稍候...</div>`;

        // 准备发送文件
        const file = fileInput.files[0];
        if (!file) {
            alert('请选择图片文件');
            return;
        }

        const formData = new FormData();
        formData.append('image', file);
        formData.append('top_k', topK);
        formData.append('threshold', threshold);

        const startTime=performance.now();

        try {
            const response = await fetch('/search', {
                method: 'POST',
                body: formData
            });

            const endTime = performance.now();
            const elapsed = ((endTime - startTime) / 1000).toFixed(2);

            const data = await response.json();

            if (data.success) {
                // 获取"显示检索耗时"复选框的状态
                const showTime = document.getElementById('showTime').checked;
                displayResults(data.results, showTime ? elapsed : null);
            } else {
                resultBox.innerHTML = `<div class="empty-result">检索失败: ${data.error || '未知错误'}</div>`;
            }
        } catch (error) {
            console.error('检索出错:', error);
            resultBox.innerHTML = `<div class="empty-result">网络错误，请稍后重试</div>`;
        }
    });
}

// 显示检索结果
function displayResults(results, elapsedTime) {
    if (!results || results.length === 0) {
        resultBox.innerHTML = `<div class="empty-result">未找到相似图像</div>`;
        return;
    }

    // 获取是否显示关键点的开关状态
    // const showKeypoints = document.getElementById('showKeypoints').checked;
    // const displayMode = document.querySelector('input[name="kp_display_mode"]:checked')?.value || 'all';

    let html = '';
    // 如果传入了耗时，就显示在顶部
    if (elapsedTime !== null) {
        html = `<div style="margin-bottom: 12px; font-size: 13px; color: #666;"> 检索耗时: ${elapsedTime} 秒</div>`;
    }

    html += '<div class="result-grid">';
    for (let i = 0; i < results.length; i++) {
        const item = results[i];
        const imgUrl = item.image_url;
        html += `
            <div class="result-card" data-img-url="${imgUrl}" data-kp-url="${item.keypoints_url || ''}" data-matches='${JSON.stringify(item.matches || [])}'>
                <img src="${imgUrl}" alt="${item.filename}">
                <div class="score-desc">
                    第${i+1}名 | 相似度 ${(item.score * 100).toFixed(1)}%<br>
                    <span style="font-size: 11px; color: #999;">匹配特征点: ${item.inliers}</span>
                </div>
            </div>
        `;
    }
    html += '</div>';
    resultBox.innerHTML = html;

    // 绑定点击事件（不管开关状态，都绑定）
    document.querySelectorAll('.result-card').forEach(card => {
        card.style.cursor = 'pointer';
        card.addEventListener('click', async function() {
            // 在点击时检查开关状态
            const showKeypoints = document.getElementById('showKeypoints').checked;
            
            if (!showKeypoints) {
                // 没勾选就不显示关键点，直接显示大图
                const imgUrl = this.dataset.imgUrl;
                showImage(imgUrl);
                return;
            }
            
            // 勾选了，根据显示模式决定画什么
            const displayMode = document.querySelector('input[name="kp_display_mode"]:checked')?.value || 'all';
            const imgUrl = this.dataset.imgUrl;
            const kpUrl = this.dataset.kpUrl;

            const matchesRaw = this.dataset.matches;
            const matches = matchesRaw ? JSON.parse(matchesRaw) : [];
            
            if (displayMode === 'all') {
                // 显示全部关键点
                if (!kpUrl) {
                    alert('该图片没有关键点数据');
                    return;
                }
                await showImageWithAllKeypoints(imgUrl, kpUrl);
            } else if (displayMode === 'matches') {
                // 只显示匹配点
                if (!matches || matches.length === 0) {
                    alert('该图片没有匹配点数据');
                    return;
                }
                await showImageWithMatches(imgUrl, matches);
            }
        });
    });
}

// 新增：普通显示大图（不显示关键点）
function showImage(imgUrl) {
    console.log('加载图片:', imgUrl);  // 看控制台输出
    
    const modal = document.createElement('div');
    modal.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.8);z-index:10000;display:flex;justify-content:center;align-items:center';
    
    const img = document.createElement('img');
    img.style.cssText = 'max-width:90%;max-height:90%';
    img.onload = () => console.log('图片加载成功');
    img.onerror = () => {
        console.error('图片加载失败:', imgUrl);
        modal.innerHTML = '<div style="color:white">图片加载失败</div>';
    };
    img.src = imgUrl;
    
    modal.appendChild(img);
    modal.onclick = () => modal.remove();
    document.body.appendChild(modal);
}

// 显示全部关键点
async function showImageWithAllKeypoints(imgUrl, kpUrl) {
    try {
        // 加载图片和关键点
        const [img, kpData] = await Promise.all([
            loadImage(imgUrl),
            fetch(kpUrl).then(res => res.arrayBuffer())
        ]);
        
        const keypoints = new Float32Array(kpData);
        
        // 创建 Canvas 绘制
        const modal = document.createElement('div');
        modal.style.position = 'fixed';
        modal.style.top = '0';
        modal.style.left = '0';
        modal.style.width = '100%';
        modal.style.height = '100%';
        modal.style.backgroundColor = 'rgba(0,0,0,0.8)';
        modal.style.zIndex = '10000';
        modal.style.display = 'flex';
        modal.style.justifyContent = 'center';
        modal.style.alignItems = 'center';
        
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        
        // 绘制图片
        ctx.drawImage(img, 0, 0);
        
        // 绘制关键点（红色小圆点）
        ctx.fillStyle = '#ff0000';
        for (let i = 0; i < keypoints.length; i += 2) {
            const x = keypoints[i];
            const y = keypoints[i+1];
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, 2 * Math.PI);
            ctx.fill();
        }
        
        modal.appendChild(canvas);
        modal.onclick = () => modal.remove();
        document.body.appendChild(modal);
        
    } catch (error) {
        console.error('加载关键点失败:', error);
        alert('加载关键点失败');
    }
}

// 只显示匹配点
async function showImageWithMatches(imgUrl, matches) {
    try {
        const img = await loadImage(imgUrl);
        
        const modal = document.createElement('div');
        modal.style.position = 'fixed';
        modal.style.top = '0';
        modal.style.left = '0';
        modal.style.width = '100%';
        modal.style.height = '100%';
        modal.style.backgroundColor = 'rgba(0,0,0,0.8)';
        modal.style.zIndex = '10000';
        modal.style.display = 'flex';
        modal.style.justifyContent = 'center';
        modal.style.alignItems = 'center';
        
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        
        ctx.drawImage(img, 0, 0);
        
        // 画匹配点（绿色）
        ctx.fillStyle = '#00ff00';
        for (let pt of matches) {
            const [x, y] = pt.db;
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
        }
        
        // 可选：显示匹配点数量
        ctx.fillStyle = '#ffffff';
        ctx.font = '14px Arial';
        ctx.fillText(`匹配点数量: ${matches.length}`, 10, 30);
        
        modal.appendChild(canvas);
        modal.onclick = () => modal.remove();
        document.body.appendChild(modal);
        
    } catch (error) {
        console.error('显示匹配点失败:', error);
        alert('显示匹配点失败');
    }
}

function loadImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = url;
    });
}

// 当勾选"显示特征关键点"时，显示/隐藏两个单选按钮
document.getElementById('showKeypoints').addEventListener('change', function() {
    const optionsDiv = document.getElementById('kp_display_options');
    if (optionsDiv) {
        optionsDiv.style.display = this.checked ? 'block' : 'none';
    }
});

// 页面加载时，根据初始状态设置显示/隐藏
document.addEventListener('DOMContentLoaded', function() {
    const showCheckbox = document.getElementById('showKeypoints');
    const optionsDiv = document.getElementById('kp_display_options');
    if (showCheckbox && optionsDiv) {
        optionsDiv.style.display = showCheckbox.checked ? 'block' : 'none';
    }
});