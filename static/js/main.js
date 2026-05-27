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

// 防抖函数
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

// 将检索逻辑抽成独立函数
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

// 阈值滑块防抖自动检索
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

// 搜索按钮
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

// 普通显示大图（不显示关键点）
function showImage(imgUrl) {
    console.log('加载图片:', imgUrl); 
    
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
        const queryImgUrl = preImg.src;
        const [queryImg, dbImg] = await Promise.all([
            loadImage(queryImgUrl),
            loadImage(imgUrl)
        ]);
        
        // ========== 获取当前 bbox 偏移 ==========
        const currentBbox = bbox;  // 全局变量
        const offsetX = currentBbox ? currentBbox[0] : 0;
        const offsetY = currentBbox ? currentBbox[1] : 0;
        console.log('bbox 偏移:', offsetX, offsetY);
        
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
        
        // 固定显示高度
        const maxHeight = 600;
        const scaleQuery = maxHeight / queryImg.height;
        const scaleDb = maxHeight / dbImg.height;
        
        const displayWidthQuery = queryImg.width * scaleQuery;
        const displayWidthDb = dbImg.width * scaleDb;
        const displayHeight = maxHeight;
        
        const gap = 60;
        const canvas = document.createElement('canvas');
        canvas.width = displayWidthQuery + displayWidthDb + gap;
        canvas.height = displayHeight;
        const ctx = canvas.getContext('2d');
        
        // 绘制查询图（左侧）
        ctx.drawImage(queryImg, 0, 0, displayWidthQuery, displayHeight);
        
        // 绘制匹配图（右侧）
        ctx.drawImage(dbImg, displayWidthQuery + gap, 0, displayWidthDb, displayHeight);
        
        // 绘制连线
        ctx.beginPath();
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 1.5;
        
        let matchCount = 0;
        let drawnCount = 0;
        for (let pt of matches) {
            // ========== 关键：查询图坐标加上 bbox 偏移 ==========
            let qx = (pt.query[0] + offsetX) * scaleQuery;
            let qy = (pt.query[1] + offsetY) * scaleQuery;
            
            let dx = pt.db[0] * scaleDb + displayWidthQuery + gap;
            let dy = pt.db[1] * scaleDb;

            const isValid = qx >= 0 && qx <= displayWidthQuery && 
                            qy >= 0 && qy <= displayHeight &&
                            dx >= displayWidthQuery + gap && dx <= canvas.width &&
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
        
        // 绘制关键点（红色，查询图）
        ctx.fillStyle = '#ff0000';
        for (let pt of matches.slice(0, 100)) {
            // ========== 关键：查询图坐标加上 bbox 偏移 ==========
            let x = (pt.query[0] + offsetX) * scaleQuery;
            let y = (pt.query[1] + offsetY) * scaleQuery;
            if (x >= 0 && x <= displayWidthQuery && y >= 0 && y <= displayHeight) {
                ctx.beginPath();
                ctx.arc(x, y, 2, 0, 2 * Math.PI);
                ctx.fill();
            }
        }
        
        // 绘制关键点（绿色，匹配图）
        ctx.fillStyle = '#00ff00';
        for (let pt of matches.slice(0, 100)) {
            let x = pt.db[0] * scaleDb + displayWidthQuery + gap;
            let y = pt.db[1] * scaleDb;
            if (x >= displayWidthQuery + gap && x <= canvas.width && y >= 0 && y <= displayHeight) {
                ctx.beginPath();
                ctx.arc(x, y, 2, 0, 2 * Math.PI);
                ctx.fill();
            }
        }
        
        // 添加文字标注
        ctx.fillStyle = '#ffffff';
        ctx.font = '16px Arial';
        ctx.fillText('查询图', 10, 30);
        ctx.fillText('匹配结果', displayWidthQuery + gap + 10, 30);
        ctx.font = '12px Arial';
        ctx.fillStyle = '#00ff00';
        ctx.fillText(`匹配点: ${matches.length} 个`, displayWidthQuery + gap + 10, 60);
        
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

// ========== Bounding Box 框选功能 ==========
let bbox = null;  // 存储 [xmin, ymin, xmax, ymax]
let isDrawing = false;
let startX, startY;

const previewArea = document.getElementById('previewArea');
const previewCanvas = document.getElementById('previewCanvas');
const clearBboxBtn = document.getElementById('clearBboxBtn');
const bboxStatus = document.getElementById('bboxStatus');

// 图片上传后，初始化 Canvas
if (fileInput) {
    const originalFileInput = fileInput.cloneNode(true);
    fileInput.addEventListener('change', function(e) {
        let file = e.target.files[0];
        if (!file) return;
        
        let url = URL.createObjectURL(file);
        preImg.src = url;
        preImg.style.display = 'block';
        emptyTip.style.display = 'none';
        
        // 等待图片加载完成后初始化 Canvas
        preImg.onload = function() {
            initCanvas();
        };
    });
}

function initCanvas() {
    // 获取显示区域尺寸
    const containerRect = previewArea.getBoundingClientRect();
    const maxWidth = containerRect.width - 20;
    const maxHeight = containerRect.height - 20;
    
    // 计算缩放比例
    const scale = Math.min(maxWidth / preImg.naturalWidth, maxHeight / preImg.naturalHeight);
    const displayWidth = preImg.naturalWidth * scale;
    const displayHeight = preImg.naturalHeight * scale;
    
    previewCanvas.width = preImg.naturalWidth;
    previewCanvas.height = preImg.naturalHeight;
    previewCanvas.style.width = displayWidth + 'px';
    previewCanvas.style.height = displayHeight + 'px';
    previewCanvas.style.position = 'absolute';
    previewCanvas.style.top = '50%';
    previewCanvas.style.left = '50%';
    previewCanvas.style.transform = 'translate(-50%, -50%)';
    
    previewCanvas.style.display = 'block';
    preImg.style.visibility = 'hidden';
    
    const ctx = previewCanvas.getContext('2d');
    ctx.drawImage(preImg, 0, 0, previewCanvas.width, previewCanvas.height);
    
    bindCanvasEvents();
}

function bindCanvasEvents() {
    previewCanvas.addEventListener('mousedown', onMouseDown);
    previewCanvas.addEventListener('mousemove', onMouseMove);
    previewCanvas.addEventListener('mouseup', onMouseUp);
}
function onMouseUp(e) {
    if (!isDrawing) return;
    isDrawing = false;
    
    const rect = previewCanvas.getBoundingClientRect();
    const scaleX = previewCanvas.width / rect.width;
    const scaleY = previewCanvas.height / rect.height;
    
    const endX = (e.clientX - rect.left) * scaleX;
    const endY = (e.clientY - rect.top) * scaleY;
    
    // 计算 bbox
    const xmin = Math.min(startX, endX);
    const ymin = Math.min(startY, endY);
    const xmax = Math.max(startX, endX);
    const ymax = Math.max(startY, endY);
    
    // 框至少要有 5 像素
    if (xmax - xmin > 5 && ymax - ymin > 5) {
        bbox = [Math.floor(xmin), Math.floor(ymin), Math.floor(xmax), Math.floor(ymax)];
        bboxStatus.innerText = `已框选区域: (${bbox[0]}, ${bbox[1]}) → (${bbox[2]}, ${bbox[3]})`;
        clearBboxBtn.style.display = 'inline-block';
        
        // 绘制最终框（绿色）
        drawFinalBbox();
    } else {
        // 框太小，清除临时框
        redrawCanvas();
    }
}

function drawFinalBbox() {
    const ctx = previewCanvas.getContext('2d');
    ctx.drawImage(preImg, 0, 0, previewCanvas.width, previewCanvas.height);
    
    if (bbox) {
        ctx.strokeStyle = '#00ff00';
        ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
        ctx.fillRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
    }
}

function onMouseDown(e) {
    const rect = previewCanvas.getBoundingClientRect();
    const scaleX = previewCanvas.width / rect.width;
    const scaleY = previewCanvas.height / rect.height;
    
    startX = (e.clientX - rect.left) * scaleX;
    startY = (e.clientY - rect.top) * scaleY;
    isDrawing = true;
}

function onMouseMove(e) {
    if (!isDrawing) return;
    
    const rect = previewCanvas.getBoundingClientRect();
    const scaleX = previewCanvas.width / rect.width;
    const scaleY = previewCanvas.height / rect.height;
    
    const currentX = (e.clientX - rect.left) * scaleX;
    const currentY = (e.clientY - rect.top) * scaleY;
    
    // 改为调用 drawTempRect，直接画临时框
    drawTempRect(currentX, currentY);
}

function drawTempRect(currentX, currentY) {
    const ctx = previewCanvas.getContext('2d');
    // 重绘原图
    ctx.drawImage(preImg, 0, 0, previewCanvas.width, previewCanvas.height);
    // 绘制已有 bbox
    if (bbox) {
        ctx.strokeStyle = '#00ff00';
        ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
        ctx.lineWidth = 2;
        ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
        ctx.fillRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
    }
    // 绘制临时框（红色虚线）
    ctx.strokeStyle = '#ff3333';
    ctx.fillStyle = 'rgba(255, 51, 51, 0.1)';  // 浅红色半透明填充
    ctx.lineWidth = 3;
    ctx.setLineDash([]);  // 改成实线，更明显
    ctx.strokeRect(startX, startY, currentX - startX, currentY - startY);
    ctx.fillRect(startX, startY, currentX - startX, currentY - startY);
}

function redrawCanvas(currentX = null, currentY = null) {
    const ctx = previewCanvas.getContext('2d');
    // 重绘原图
    ctx.drawImage(preImg, 0, 0, previewCanvas.width, previewCanvas.height);
    
    // 绘制已有 bbox
    if (bbox) {
        ctx.strokeStyle = '#00ff00';
        ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
        ctx.lineWidth = 2;
        ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
        ctx.fillRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
    }
    
    // 绘制临时框
    if (currentX !== null && currentY !== null && startX !== undefined) {
        ctx.strokeStyle = '#ff0000';
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(startX, startY, currentX - startX, currentY - startY);
        ctx.setLineDash([]);
    }
}

function drawBbox() {
    redrawCanvas();
}

// 清除框选
if (clearBboxBtn) {
    clearBboxBtn.addEventListener('click', function() {
        bbox = null;
        bboxStatus.innerText = '';
        clearBboxBtn.style.display = 'none';
        redrawCanvas();
    });
}

// 修改 performSearch 函数，添加 bbox 参数
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
    
    // 添加 bbox（如果有）
    if (bbox) {
        formData.append('bbox', JSON.stringify(bbox));
    }

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