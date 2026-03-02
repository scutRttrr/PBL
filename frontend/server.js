const express = require('express');
const app = express();
const path = require('path');

app.use(express.json());

// 首页：展示输入框
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(3000, () => {
    console.log('前端测试页面已启动: http://localhost:3000');
});