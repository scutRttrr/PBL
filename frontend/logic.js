// --- 1. ILO 增删逻辑 ---
function addILO() {
    const container = document.getElementById('ilo-container');
    const div = document.createElement('div');
    div.className = 'ilo-item';
    div.innerHTML = `
        <input type="text" class="ilo-input" placeholder="请输入教学目标">
        <button class="btn btn-delete" onclick="removeILO(this)">×</button>
    `;
    container.appendChild(div);
    div.querySelector('input').focus();
}

function removeILO(btn) {
    const container = document.getElementById('ilo-container');
    if (container.children.length > 1) {
        btn.parentElement.remove();
    }
}

// --- 2. 提交逻辑 ---
async function submitRequest() {
    const submitBtn = document.getElementById('submit_btn');
    const responseContainer = document.getElementById('response-container');
    const responseContent = document.getElementById('response-content');
    const cursor = document.getElementById('cursor');

    const iloElements = document.querySelectorAll('.ilo-input');
    const questionsList = Array.from(iloElements)
        .map(input => input.value.trim())
        .filter(val => val !== "");

    if (questionsList.length === 0) {
        alert("请至少填写一个教学目标 (ILO)");
        return;
    }

    const payload = {
        questions: questionsList,
        course_info: document.getElementById('course_input').value.trim() || null,
        class_info: document.getElementById('class_input').value.trim() || null
    };

    // 重置状态
    submitBtn.disabled = true;
    submitBtn.innerText = "🚀 正在生成中...";
    responseContent.innerText = "";
    cursor.style.display = 'inline-block';

    try {
        const response = await fetch('http://127.0.0.1:8000/generate-tc', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error(`服务器响应异常: ${response.status}`);
        const threadId = response.headers.get('Task-ID');
        if (threadId) localStorage.setItem('Task-ID', threadId);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        console.log("获取到task id" + threadId);

        while (true) {
            const { done, value } = await reader.read();
            if (done) {
            // 1. 获取本次请求的 Task-ID (之前你应该已经存入 threadId 了)
            const threadId = response.headers.get('Task-ID');
            if (threadId) {
                localStorage.setItem('Task-ID', threadId); // 确保 ID 名字与详情页一致
            }

            // 2. 直接跳转到详情页面
            window.location.href = 'revision.html';
            break;
            }

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');
            for (const line of lines) {
                if (line.trim().startsWith('data: ')) {
                    const jsonStr = line.replace('data: ', '').trim();
                    try {
                        const data = JSON.parse(jsonStr);
                        if (data.content) {
                            responseContent.innerText += data.content;
                            responseContainer.scrollTop = responseContainer.scrollHeight;
                        }
                    } catch (e) { console.log("解析碎块失败", e); }
                }
            }
        }
    } catch (err) {
        responseContent.innerText = "❌ 发生错误: " + err.message;
        responseContent.style.color = "var(--danger-red)";
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerText = "生成教案";
        cursor.style.display = 'none';
    }
}

// --- 3. 视图切换逻辑 (独立出来，不要写在 submitRequest 里面) ---
function switchToDetailView() {
    const form = document.getElementById('form-container');
    const detail = document.getElementById('detail-container');

    if (form && detail) {
        form.style.display = 'none';
        detail.style.display = 'block';
        fetchFinalPlan();
    } else {
        // 如果报错，说明你的 HTML 里缺少对应的 ID
        alert("页面容器加载失败，请检查 HTML 中是否存在 form-container 和 detail-container");
    }
}

// --- 4. 获取最终教案 ---
async function fetchFinalPlan() {
    const detailBox = document.getElementById('final-plan-content');
    const taskId = localStorage.getItem('Task-ID');

    if (!taskId) return;

    try {
        const response = await fetch(`http://127.0.0.1:8000/get-final-plan?task_id=${taskId}`);
        if (!response.ok) throw new Error("获取教案失败");
        const data = await response.json();
        detailBox.innerHTML = `<div style="padding:20px; background:#f9f9f9; border-radius:8px;">${data.full_plan}</div>`;
    } catch (err) {
        detailBox.innerHTML = `<div style="color:red;">❌ 加载失败: ${err.message}</div>`;
    }
}