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

// ========== 新增：防抖函数 ==========
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ========== 新增：将检索逻辑抽成独立函数 ==========
async function performSearch() {
    if (!preImg.src || preImg.style.display === 'none') {
        alert('请先上传图片');
        return;
    }

    const topK = sliderN ? parseInt(sliderN.value) : 4;
    const threshold = sliderTh ? parseFloat(sliderTh.value) : 5.0;

    resultBox.innerHTML = `<div class="empty-result">检索中，请稍候...</div>`;

    const file = fileInput.files[0];
    if (!file) {
        alert('请选择图片文件');
        return;
    }

    const formData = new FormData();
    formData.append('image', file);
    formData.append('top_k', topK);
    formData.append('threshold', threshold);

    const startTime = performance.now();

    try {
        const response = await fetch('/search', {
            method: 'POST',
            body: formData
        });

        const endTime = performance.now();
        const elapsed = ((endTime - startTime) / 1000).toFixed(2);

        const data = await response.json();

        if (data.success) {
            const showTime = document.getElementById('showTime').checked;
            displayResults(data.results, showTime ? elapsed : null);
        } else {
            resultBox.innerHTML = `<div class="empty-result">检索失败: ${data.error || '未知错误'}</div>`;
        }
    } catch (error) {
        console.error('检索出错:', error);
        resultBox.innerHTML = `<div class="empty-result">网络错误，请稍后重试</div>`;
    }
}

// ========== 新增：阈值滑块防抖自动检索 ==========
if (sliderTh) {
    const debounceSearch = debounce(performSearch, 500);
    sliderTh.oninput = function() {
        valTh.innerText = this.value;
        // 如果已经有检索结果，自动重新检索
        if (resultBox.innerHTML && !resultBox.innerHTML.includes('检索中') && preImg.src && preImg.style.display !== 'none') {
            debounceSearch();
        }
    };
}

// 修改搜索按钮，复用 performSearch
if (searchBtn) {
    searchBtn.addEventListener('click', async () => {
        await performSearch();
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
        const currentTh = sliderTh ? parseFloat(sliderTh.value) : 5.0;
        html = `<div style="margin-bottom: 12px; font-size: 13px; color: #666;"> 检索耗时: ${elapsedTime} 秒 | 当前阈值: ${currentTh.toFixed(1)} 像素</div>`;
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

// 只显示匹配点（带连线）
async function showImageWithMatches(imgUrl, matches) {
    try {
        // 同时加载查询图和匹配图
        const queryImgUrl = preImg.src;  // 用户上传的查询图
        const [queryImg, dbImg] = await Promise.all([
            loadImage(queryImgUrl),
            loadImage(imgUrl)
        ]);
        
        // 创建模态框
        const modal = document.createElement('div');
        modal.style.position = 'fixed';
        modal.style.top = '0';
        modal.style.left = '0';
        modal.style.width = '100%';
        modal.style.height = '100%';
        modal.style.backgroundColor = 'rgba(0,0,0,0.9)';
        modal.style.zIndex = '10000';
        modal.style.display = 'flex';
        modal.style.justifyContent = 'center';
        modal.style.alignItems = 'center';
        
        // 计算合适的显示尺寸（两张图并排，各占一半宽度）
        const maxWidth = Math.min(window.innerWidth * 0.4, queryImg.width);
        const maxHeight = window.innerHeight * 0.7;
        
        const scale = Math.min(maxWidth / queryImg.width, maxHeight / queryImg.height);
        const displayWidth = queryImg.width * scale;
        const displayHeight = queryImg.height * scale;
        
        // 创建 Canvas（宽度为两张图宽度之和 + 间距）
        const gap = 60;  // 两张图之间的间距
        const canvas = document.createElement('canvas');
        canvas.width = displayWidth * 2 + gap;
        canvas.height = displayHeight;
        const ctx = canvas.getContext('2d');
        
        // 绘制查询图（左侧）
        ctx.drawImage(queryImg, 0, 0, displayWidth, displayHeight);
        
        // 绘制匹配图（右侧）
        ctx.drawImage(dbImg, displayWidth + gap, 0, displayWidth, displayHeight);
        
        // 绘制连线（绿色线条）
        ctx.beginPath();
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 1.5;
        
        let matchCount = 0;
        let drawnCount = 0;
        for (let pt of matches) {
            // 获取查询图关键点坐标（需要缩放）
            let qx = pt.query[0] * scale;
            let qy = pt.query[1] * scale;
            
            // 获取匹配图关键点坐标（需要缩放 + 右移）
            let dx = pt.db[0] * scale + displayWidth + gap;
            let dy = pt.db[1] * scale;

            // 调试：检查坐标是否在画布范围内
            const isValid = qx >= 0 && qx <= displayWidth && 
                            qy >= 0 && qy <= displayHeight &&
                            dx >= displayWidth + gap && dx <= canvas.width &&
                            dy >= 0 && dy <= displayHeight;
            
            if (!isValid) {
                console.log(`点 ${matchCount} 超出范围: query(${qx},${qy}) db(${dx},${dy})`);
            }
            
            // 只画前 50 条连线，避免太密集
            if (matchCount < 50 && isValid) {
                ctx.beginPath();
                ctx.moveTo(qx, qy);
                ctx.lineTo(dx, dy);
                ctx.stroke();
                drawnCount++;
            }
            matchCount++;
        }
        console.log(`总匹配点: ${matchCount}, 实际画线: ${drawnCount}`);
        
        // 绘制关键点（查询图上画红色，匹配图上画绿色）
        // 查询图上的关键点（红色）
        ctx.fillStyle = '#ff0000';
        for (let pt of matches.slice(0, 100)) {
            let x = pt.query[0] * scale;
            let y = pt.query[1] * scale;
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, 2 * Math.PI);
            ctx.fill();
        }
        
        // 匹配图上的关键点（绿色）
        ctx.fillStyle = '#00ff00';
        for (let pt of matches.slice(0, 100)) {
            let x = pt.db[0] * scale + displayWidth + gap;
            let y = pt.db[1] * scale;
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, 2 * Math.PI);
            ctx.fill();
        }
        
        // 添加文字标注
        ctx.fillStyle = '#ffffff';
        ctx.font = '16px Arial';
        ctx.fillText('查询图', 10, 30);
        ctx.fillText('匹配结果', displayWidth + gap + 10, 30);
        
        // 显示匹配点数量
        ctx.font = '12px Arial';
        ctx.fillStyle = '#00ff00';
        ctx.fillText(`匹配点: ${matches.length} 个`, displayWidth + gap + 10, 60);
        
        modal.appendChild(canvas);
        
        // 添加关闭提示
        const closeHint = document.createElement('div');
        closeHint.style.position = 'absolute';
        closeHint.style.bottom = '20px';
        closeHint.style.left = '50%';
        closeHint.style.transform = 'translateX(-50%)';
        closeHint.style.color = '#888';
        closeHint.style.fontSize = '12px';
        closeHint.style.backgroundColor = 'rgba(0,0,0,0.6)';
        closeHint.style.padding = '5px 12px';
        closeHint.style.borderRadius = '20px';
        closeHint.innerText = '点击任意位置关闭';
        modal.appendChild(closeHint);
        
        modal.onclick = () => modal.remove();
        document.body.appendChild(modal);
        
    } catch (error) {
        console.error('显示匹配点失败:', error);
        alert('显示匹配点失败: ' + error.message);
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