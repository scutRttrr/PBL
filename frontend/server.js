const express = require('express');
const app = express();
const path = require('path');

app.use(express.json());

// 1. 核心修复：托管当前目录下的所有静态文件 (js, css, html)
// 这样访问 localhost:3000/detail.html 就能打开详情页了
app.use(express.static(__dirname));

// 2. 首页路由
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

const PORT = 3000;
app.listen(PORT, () => {
    // 如果没有这一行，终端就不会有任何输出
    console.log(`✅ 服务器启动成功：http://localhost:${PORT}`);
});