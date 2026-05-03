// 全局变量
let historyData = [];

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 加载历史记录
    loadHistory();
    
    // 绑定表单提交事件
    document.getElementById('upload-form').addEventListener('submit', function(e) {
        e.preventDefault();
        uploadDocument();
    });
});

// 上传文档
function uploadDocument() {
    const fileInput = document.getElementById('document-upload');
    const file = fileInput.files[0];
    const statusElement = document.getElementById('upload-status');
    const resultContent = document.getElementById('result-content');
    
    // 检查文件是否存在
    if (!file) {
        showStatus('请选择一个文件', 'error');
        return;
    }
    
    // 显示上传状态
    showStatus('正在上传和处理文档...', 'success');
    resultContent.innerHTML = '';
    
    // 创建FormData对象
    const formData = new FormData();
    formData.append('file', file);
    
    // 模拟API调用（实际项目中替换为真实API）
    setTimeout(() => {
        // 模拟分类结果
        const mockResult = {
            document_id: 'doc_' + Date.now(),
            filename: file.name,
            predicted_class: '通知_公告',
            confidence: 0.95,
            ocr_regions: [
                { text: '关于召开会议的通知', confidence: 0.98, box: [100, 50, 300, 80], region_type: 'title' },
                { text: '各部门：\n为了总结工作，部署下一阶段任务，现决定召开全体会议。\n时间：2024年12月31日\n地点：会议室', confidence: 0.96, box: [100, 100, 400, 200], region_type: 'text' }
            ]
        };
        
        // 显示结果
        displayResult(mockResult);
        
        // 保存到历史记录
        saveToHistory(mockResult);
        
        // 显示成功状态
        showStatus('文档处理完成！', 'success');
    }, 2000);
    
    // 实际API调用示例
    /*
    fetch('http://localhost:8000/api/v1/classification/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showStatus('处理失败：' + data.error, 'error');
        } else {
            displayResult(data);
            saveToHistory(data);
            showStatus('文档处理完成！', 'success');
        }
    })
    .catch(error => {
        showStatus('处理失败：' + error.message, 'error');
    });
    */
}

// 显示状态信息
function showStatus(message, type) {
    const statusElement = document.getElementById('upload-status');
    statusElement.textContent = message;
    statusElement.className = 'status ' + type;
}

// 显示分类结果
function displayResult(result) {
    const resultContent = document.getElementById('result-content');
    
    let html = `
        <div class="result-item">
            <strong>文件名：</strong>${result.filename}<br>
            <strong>预测类别：</strong><span class="class">${result.predicted_class}</span><br>
            <strong>置信度：</strong><span class="confidence">${(result.confidence * 100).toFixed(2)}%</span>
        </div>
    `;
    
    // 显示OCR结果
    if (result.ocr_regions && result.ocr_regions.length > 0) {
        html += '<div class="result-item"><strong>OCR提取内容：</strong><ul>';

        
        result.ocr_regions.forEach(region => {
            html += `
                <li>
                    <strong>${region.region_type}：</strong>${region.text}
                    <small>(置信度：${(region.confidence * 100).toFixed(2)}%)</small>
                </li>
            `;
        });
        
        html += '</ul></div>';
    }
    
    resultContent.innerHTML = html;
}

// 保存到历史记录
function saveToHistory(result) {
    // 添加时间戳
    result.timestamp = new Date().toISOString();
    
    // 添加到历史数据
    historyData.unshift(result);
    
    // 限制历史记录数量
    if (historyData.length > 10) {
        historyData = historyData.slice(0, 10);
    }
    
    // 保存到本地存储
    localStorage.setItem('docClassifyHistory', JSON.stringify(historyData));
    
    // 更新历史记录显示
    displayHistory();
}

// 加载历史记录
function loadHistory() {
    const savedHistory = localStorage.getItem('docClassifyHistory');
    if (savedHistory) {
        historyData = JSON.parse(savedHistory);
        displayHistory();
    }
}

// 显示历史记录
function displayHistory() {
    const historyList = document.getElementById('history-list');
    
    if (historyData.length === 0) {
        historyList.innerHTML = '<div class="history-item">暂无历史记录</div>';
        return;
    }
    
    let html = '';
    historyData.forEach(item => {
        const date = new Date(item.timestamp).toLocaleString();
        html += `
            <div class="history-item">
                <div class="filename">${item.filename}</div>
                <div>类别：<span class="class">${item.predicted_class}</span></div>
                <div>置信度：<span class="confidence">${(item.confidence * 100).toFixed(2)}%</span></div>
                <div class="timestamp">${date}</div>
            </div>
        `;
    });
    
    historyList.innerHTML = html;
}