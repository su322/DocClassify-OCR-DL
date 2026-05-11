const API_BASE = 'http://127.0.0.1:8000/api/v1';

const CLASS_NAMES_CN = {
    'letter': '信件 (letter)',
    'form': '表单 (form)',
    'email': '电子邮件 (email)',
    'handwritten': '手写文稿 (handwritten)',
    'advertisement': '广告 (advertisement)',
    'scientific_report': '科学报告 (scientific_report)',
    'scientific_publication': '科学出版物 (scientific_publication)',
    'specification': '规范文档 (specification)',
    'file_folder': '文件夹 (file_folder)',
    'news_article': '新闻文章 (news_article)',
    'budget': '预算 (budget)',
    'invoice': '发票 (invoice)',
    'presentation': '演示文稿 (presentation)',
    'questionnaire': '问卷 (questionnaire)',
    'resume': '简历 (resume)',
    'memo': '备忘录 (memo)',
};

const LAYOUT_TYPES_CN = {
    'text': '正文', 'title': '标题', 'figure': '图片', 'caption': '图注',
    'header': '页眉', 'footer': '页脚', 'table': '表格', 'reference': '参考文献',
    'equation': '公式', 'general': '其他',
};

let historyData = [];

document.addEventListener('DOMContentLoaded', async function () {
    await loadModelList();
    loadHistory();
    document.getElementById('upload-form').addEventListener('submit', async function (e) {
        e.preventDefault();
        await uploadDocument();
    });
});

async function loadModelList() {
    const sel = document.getElementById('model-select');
    sel.innerHTML = '<option value="">加载模型中...</option>';
    sel.disabled = true;
    try {
        const resp = await fetch(`${API_BASE}/classification/models`);
        const body = await resp.json();
        const models = body.data || [];
        sel.innerHTML = models.map(m =>
            `<option value="${m.model_id}" ${m.model_id === 'gcn_reading_order' ? 'selected' : ''}>
                ${m.name}
            </option>`
        ).join('');
        sel.disabled = false;
    } catch (e) {
        sel.innerHTML = '<option value="gcn_reading_order">GCN (阅读顺序) — 默认</option>';
        console.warn('无法加载模型列表，使用默认选项', e);
    }
}

async function uploadDocument() {
    const fileInput = document.getElementById('document-upload');
    const file = fileInput.files[0];
    const statusElement = document.getElementById('upload-status');
    const resultContent = document.getElementById('result-content');
    const submitBtn = document.getElementById('submit-btn');

    if (!file) {
        showStatus('请选择一个文件', 'error');
        return;
    }

    const modelId = document.getElementById('model-select').value;

    showStatus('正在上传和处理文档...', 'loading');
    resultContent.innerHTML = '<div class="loading-spinner"></div>';
    submitBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_id', modelId);

    try {
        const response = await fetch(`${API_BASE}/ocr/classify?model_id=${modelId}`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: '请求失败' }));
            throw new Error(err.detail || `HTTP ${response.status}`);
        }

        const body = await response.json();
        const result = body.data;

        displayResult(result, file.name);
        saveToHistory(result, file.name);
        showStatus('文档处理完成！', 'success');
    } catch (error) {
        showStatus('处理失败：' + error.message, 'error');
        resultContent.innerHTML = `<div class="error-message">${error.message}</div>`;
    } finally {
        submitBtn.disabled = false;
    }
}

function showStatus(message, type) {
    const el = document.getElementById('upload-status');
    el.textContent = message;
    el.className = 'status ' + type;
}

function displayResult(result, filename) {
    const el = document.getElementById('result-content');
    const cls = result.predicted_class;
    const cnName = CLASS_NAMES_CN[cls] || cls;
    const conf = (result.confidence * 100).toFixed(2);

    let regionsHtml = '';
    if (result.ocr_regions && result.ocr_regions.length > 0) {
        result.ocr_regions.slice(0, 20).forEach((r, i) => {
            const typeCn = LAYOUT_TYPES_CN[r.region_type] || r.region_type || '未知';
            regionsHtml += `
                <div class="ocr-region">
                    <span class="region-tag">${typeCn}</span>
                    <span class="region-text">${escapeHtml(r.text)}</span>
                    <span class="region-conf">${(r.confidence * 100).toFixed(1)}%</span>
                </div>
            `;
        });
        if (result.ocr_regions.length > 20) {
            regionsHtml += `<div class="ocr-more">... 还有 ${result.ocr_regions.length - 20} 个区域</div>`;
        }
    }

    el.innerHTML = `
        <div class="result-header">
            <div class="result-icon ${conf >= 80 ? 'high' : conf >= 60 ? 'mid' : 'low'}">${cls.charAt(0).toUpperCase()}</div>
            <div class="result-summary">
                <div class="result-class">${cnName}</div>
                <div class="result-file">${escapeHtml(filename)}</div>
            </div>
        </div>
        <div class="confidence-bar-wrap">
            <div class="confidence-label">置信度</div>
            <div class="confidence-bar">
                <div class="confidence-fill ${conf >= 80 ? 'high' : conf >= 60 ? 'mid' : 'low'}" style="width:${conf}%"></div>
            </div>
            <div class="confidence-value">${conf}%</div>
        </div>
        ${regionsHtml ? `<div class="regions-section"><h3>OCR 识别区域</h3>${regionsHtml}</div>` : ''}
    `;
}

function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}

function saveToHistory(result, filename) {
    const entry = {
        document_id: result.document_id,
        filename: filename,
        predicted_class: result.predicted_class,
        confidence: result.confidence,
        timestamp: new Date().toISOString(),
    };
    historyData.unshift(entry);
    if (historyData.length > 20) historyData = historyData.slice(0, 20);
    localStorage.setItem('docClassifyHistory', JSON.stringify(historyData));
    displayHistory();
}

function loadHistory() {
    const saved = localStorage.getItem('docClassifyHistory');
    if (saved) {
        historyData = JSON.parse(saved);
        displayHistory();
    }
}

function displayHistory() {
    const el = document.getElementById('history-list');
    if (historyData.length === 0) {
        el.innerHTML = '<div class="history-empty">暂无历史记录</div>';
        return;
    }
    el.innerHTML = historyData.map(item => {
        const cls = CLASS_NAMES_CN[item.predicted_class] || item.predicted_class;
        const date = new Date(item.timestamp).toLocaleString();
        const conf = (item.confidence * 100).toFixed(1);
        return `
            <div class="history-item">
                <div class="h-filename">${escapeHtml(item.filename)}</div>
                <div class="h-meta">
                    <span class="h-class">${cls}</span>
                    <span class="h-conf">${conf}%</span>
                    <span class="h-time">${date}</span>
                </div>
            </div>
        `;
    }).join('');
}
