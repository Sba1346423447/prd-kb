/**
 * 企业技术支持智研知识库（PRD-KB）—— 前端交互脚本
 *
 * 核心能力：登录鉴权（JWT）、SSE 流式对话、Markdown 渲染、思考过程可视化。
 * 依赖：marked.js（Markdown 解析库）。
 */

/* ========== DOM 元素引用 ========== */
const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const stopBtn = document.getElementById('stopBtn');
const toast = document.getElementById('toast');
const imagePickerBtn = document.getElementById('imagePickerBtn');
const imageInput = document.getElementById('imageInput');
const imagePreviewArea = document.getElementById('imagePreviewArea');
const loginOverlay = document.getElementById('loginOverlay');
const loginForm = document.getElementById('loginForm');
const loginError = document.getElementById('loginError');
const sessionIdInput = document.getElementById('sessionId');

/* ========== 全局状态 ========== */
let isStreaming = false;
let abortController = null;
let selectedImages = [];

const MAX_IMAGES = 4;

/* ========== 检索模式切换 ========== */

/**
 * 当前检索模式：agent=Agent 自主检索，pure=固定流水线直出，auto=后端规则自动选择。
 */
let chatMode = 'agent';
const MODE_LABELS = { auto: 'Auto', agent: 'Agent', pure: 'Pure' };

const modeDropdown = document.getElementById('modeDropdown');
const modeBtn = document.getElementById('modeBtn');
const modeLabel = document.getElementById('modeLabel');

/**
 * 初始化检索模式下拉交互：选择模式、点击外部自动收起
 */
function initModeDropdown() {
  modeBtn.addEventListener('click', function(e) {
    e.stopPropagation();
    modeDropdown.classList.toggle('open');
  });

  modeDropdown.querySelectorAll('.quick-item').forEach(function(item) {
    item.addEventListener('click', function() {
      if (item.classList.contains('disabled')) return;
      chatMode = item.dataset.mode;
      modeLabel.textContent = MODE_LABELS[chatMode];
      modeDropdown.querySelectorAll('.quick-item').forEach(function(i) {
        i.classList.toggle('active', i === item);
      });
      modeDropdown.classList.remove('open');
    });
  });

  document.addEventListener('click', function(e) {
    if (!modeDropdown.contains(e.target)) {
      modeDropdown.classList.remove('open');
    }
  });
}

/* ========== 认证与令牌管理 ========== */

/**
 * 读取本地存储的 JWT 访问令牌
 * @returns {string} 令牌字符串，未登录时为空串
 */
function getToken() {
  return localStorage.getItem('token') || '';
}

/**
 * 组装带 Authorization 头的请求头
 * @param {Object} extra - 额外的请求头字段
 * @returns {Object} 合并后的请求头对象
 */
function authHeaders(extra) {
  return Object.assign({ 'Authorization': 'Bearer ' + getToken() }, extra || {});
}

/**
 * 显示登录遮罩层
 * @param {string} message - 可选的提示信息（如"登录已过期"）
 */
function showLogin(message) {
  loginError.textContent = message || '';
  loginOverlay.classList.add('show');
}

/**
 * 隐藏登录遮罩层
 */
function hideLogin() {
  loginOverlay.classList.remove('show');
}

/**
 * 更新侧边栏用户名与头像显示
 * @param {string} username - 当前用户名
 */
function updateUserDisplay(username) {
  document.getElementById('userName').textContent = username || '未登录';
  document.getElementById('userAvatar').textContent = (username || 'U').charAt(0).toUpperCase();
}

/**
 * 退出登录：清除本地令牌并回到登录页
 */
function logout() {
  localStorage.removeItem('token');
  localStorage.removeItem('username');
  sessionIdInput.value = '';
  updateUserDisplay('');
  showLogin();
}

/**
 * 处理登录表单提交：请求 /auth/login 签发令牌
 */
async function handleLogin(event) {
  event.preventDefault();
  const username = document.getElementById('loginUsername').value.trim();
  const password = document.getElementById('loginPassword').value;
  if (!username || !password) {
    loginError.textContent = '请输入用户名和密码';
    return;
  }
  try {
    const resp = await fetch('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: username, password: password })
    });
    if (resp.ok) {
      const data = await resp.json();
      localStorage.setItem('token', data.access_token);
      localStorage.setItem('username', data.username);
      updateUserDisplay(data.username);
      hideLogin();
      await createNewSession();
    } else {
      loginError.textContent = '用户名或密码错误';
    }
  } catch (e) {
    loginError.textContent = '网络请求失败，请检查服务是否正常运行';
  }
}

/**
 * 页面加载时校验本地令牌：有效则恢复登录态并申请会话，401 则回到登录页
 */
async function verifyLogin() {
  if (!getToken()) {
    showLogin();
    return;
  }
  try {
    const resp = await fetch('/auth/me', { headers: authHeaders() });
    if (resp.ok) {
      const data = await resp.json();
      updateUserDisplay(data.username);
      if (!getSessionId()) {
        await createNewSession();
      }
    } else {
      logout();
    }
  } catch (e) {
    // 网络异常时不强制登出，保留本地令牌待网络恢复后重试
  }
}

/**
 * 向服务端申请新会话并写入顶栏会话输入框
 */
async function createNewSession() {
  try {
    const resp = await fetch('/sessions', { method: 'POST', headers: authHeaders() });
    if (resp.status === 401) {
      showLogin('登录已过期，请重新登录');
      return;
    }
    if (resp.ok) {
      const data = await resp.json();
      sessionIdInput.value = data.session_id;
    }
  } catch (e) {
    showToast('创建会话失败，请检查服务是否正常运行');
  }
}

/* ========== 推荐问题列表 ========== */
const suggestionList = [
  '帮我分析一下日志',
  '帮我查一下技术文档型知识库怎么搭建？',
  '元数据怎么设计？',
  '帮我统计一下这个文本有多少字'
];

/**
 * 渲染推荐问题卡片列表，绑定点击事件以一键填入并发送
 */
function renderSuggestions() {
  const suggestionsEl = document.getElementById('suggestions');
  suggestionsEl.innerHTML = suggestionList.map(function(text) {
    return '<div class="suggestion-card">' + escapeHtml(text) + '</div>';
  }).join('');
  suggestionsEl.querySelectorAll('.suggestion-card').forEach(function(card) {
    card.addEventListener('click', function() {
      userInput.value = card.textContent;
      sendMessage();
    });
  });
}

/* ========== Markdown 配置 ========== */
marked.setOptions({
  breaks: true,
  gfm: true
});

/**
 * 将 Markdown 文本转换为 HTML
 * @param {string} text - Markdown 原始文本
 * @returns {string} 渲染后的 HTML 字符串
 */
function renderMarkdown(text) {
  if (!text) return '';
  return marked.parse(text);
}

/* ========== 工具函数 ========== */

/**
 * HTML 实体转义，防止 XSS 注入
 * @param {string} text - 原始文本
 * @returns {string} 转义后的安全文本
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * 获取当前会话 ID（由服务端创建，输入框只读展示）
 * @returns {string} 会话 ID，未创建时返回空串
 */
function getSessionId() {
  return sessionIdInput.value.trim();
}

/**
 * 显示 Toast 提示，2 秒后自动消失
 * @param {string} message - 提示文本
 */
function showToast(message) {
  toast.textContent = message;
  toast.classList.add('show');
  setTimeout(function() {
    toast.classList.remove('show');
  }, 2000);
}

/**
 * 将聊天容器滚动到底部
 */
function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

/* ========== 消息渲染 ========== */

/**
 * 向聊天容器中添加一条消息气泡
 * @param {string} role - 消息角色，user 或 assistant
 * @param {string} content - 消息内容
 * @param {boolean} isStreamingMsg - 是否为流式消息，true 时显示光标闪烁动画
 * @returns {HTMLElement} 创建的消息 DOM 元素
 */
function addMessage(role, content, isStreamingMsg, imageDataUrls) {
  const welcomeEl = document.getElementById('welcome');
  if (welcomeEl) welcomeEl.remove();
  const msgDiv = document.createElement('div');
  msgDiv.className = 'message ' + role;

  if (role === 'assistant') {
    const thinkingId = 'thinking-' + Date.now();
    msgDiv.innerHTML =
      '<div class="avatar">AI</div>' +
      '<div class="message-body">' +
        '<div class="thinking" id="' + thinkingId + '">' +
          '<div class="thinking-header">' +
            '<span class="thinking-dot"></span>' +
            '<span class="thinking-label">思考中...</span>' +
          '</div>' +
          '<div class="thinking-steps"></div>' +
        '</div>' +
        '<div class="bubble-wrapper">' +
          '<button class="copy-btn" title="复制">\u2398</button>' +
          '<div class="bubble' + (isStreamingMsg ? ' streaming' : '') + '">' +
            (isStreamingMsg ? '<span class="cursor-blink"></span>' : renderMarkdown(content)) +
          '</div>' +
        '</div>' +
      '</div>';

    const copyBtn = msgDiv.querySelector('.copy-btn');
    copyBtn.addEventListener('click', function(e) {
      e.stopPropagation();
      const bubbleEl = msgDiv.querySelector('.bubble');
      const textToCopy = bubbleEl.textContent || '';
      navigator.clipboard.writeText(textToCopy).then(function() {
        showToast('已复制到剪贴板');
      }).catch(function() {
        showToast('复制失败');
      });
    });
  } else {
    const imagesHtml = (imageDataUrls || []).map(function(src) {
      return '<div class="user-image"><img src="' + src + '" alt="用户图片"></div>';
    }).join('');
    msgDiv.innerHTML =
      '<div class="avatar">U</div>' +
      '<div class="bubble">' + imagesHtml + escapeHtml(content) + '</div>';
  }
  chatContainer.appendChild(msgDiv);
  scrollToBottom();
  return msgDiv;
}

/* ========== 思考面板交互 ========== */

/**
 * 切换思考面板的折叠/展开状态
 * @param {HTMLElement} section - 思考面板 DOM 元素
 */
function toggleThinking(section) {
  section.classList.toggle('thinking-collapsed');
}

/* ========== 流式对话核心 ========== */

/**
 * 发送用户消息，建立 SSE 连接并逐 token 渲染回答
 *
 * 流程：发送 POST 请求 -> 读取 SSE 流 -> 解析 thinking/token/done 事件 ->
 * 实时更新思考面板与回答气泡 -> 异常时展示错误提示。
 */
async function sendMessage() {
  const question = userInput.value.trim();
  if (!question || isStreaming) return;
  const sentImages = selectedImages.slice();

  isStreaming = true;
  sendBtn.disabled = true;
  sendBtn.style.display = 'none';
  stopBtn.style.display = 'flex';
  userInput.value = '';
  userInput.style.height = 'auto';
  clearImagePreviews();

  addMessage('user', question, false, sentImages);
  const assistantMsg = addMessage('assistant', '', true);
  const bubble = assistantMsg.querySelector('.bubble');
  const thinkingSection = assistantMsg.querySelector('.thinking');
  const thinkingSteps = assistantMsg.querySelector('.thinking-steps');
  const thinkingLabel = assistantMsg.querySelector('.thinking-label');
  let fullAnswer = '';

  abortController = new AbortController();

  try {
    const resp = await fetch('/chat/stream', {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        question: question,
        session_id: getSessionId(),
        images: sentImages,
        mode: chatMode
      }),
      signal: abortController.signal
    });

    if (resp.status === 401) {
      bubble.innerHTML = escapeHtml('登录已过期，请重新登录');
      bubble.classList.remove('streaming');
      bubble.style.color = '#e74c3c';
      thinkingSection.style.display = 'none';
      showLogin('登录已过期，请重新登录');
      return;
    }
    if (resp.status === 403) {
      bubble.innerHTML = escapeHtml('会话不存在或无权访问，请点击新对话创建会话');
      bubble.classList.remove('streaming');
      bubble.style.color = '#e74c3c';
      thinkingSection.style.display = 'none';
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (!line.startsWith('data: ')) continue;
        const dataStr = line.slice(6);
        let data;
        try {
          data = JSON.parse(dataStr);
        } catch (e) {
          continue;
        }

        if (data.thinking) {
          if (data.thinking.done) {
            const dot = thinkingSection.querySelector('.thinking-dot');
            dot.className = 'thinking-icon';
            dot.textContent = '\u2713';
            thinkingLabel.textContent = '思考过程';
            const header = thinkingSection.querySelector('.thinking-header');
            const arrow = document.createElement('span');
            arrow.className = 'thinking-arrow';
            arrow.textContent = '\u25B6';
            header.appendChild(arrow);
            header.addEventListener('click', function() {
              toggleThinking(thinkingSection);
            });
            thinkingSection.classList.add('thinking-collapsed');
          } else {
            const step = document.createElement('div');
            step.className = 'thinking-step';
            step.textContent = data.thinking.text;
            thinkingSteps.appendChild(step);
            thinkingLabel.textContent = data.thinking.text;
          }
        }

        if (data.token) {
          fullAnswer += data.token;
          bubble.innerHTML = renderMarkdown(fullAnswer) + '<span class="cursor-blink"></span>';
        }

        if (data.done) {
          bubble.innerHTML = renderMarkdown(fullAnswer);
          bubble.classList.remove('streaming');
        }

        if (data.error) {
          bubble.innerHTML = escapeHtml(data.error);
          bubble.classList.remove('streaming');
          bubble.style.color = '#e74c3c';
          thinkingSection.style.display = 'none';
        }
      }
      scrollToBottom();
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      if (fullAnswer) {
        bubble.innerHTML = renderMarkdown(fullAnswer);
      } else {
        bubble.innerHTML = escapeHtml('已停止生成');
      }
      bubble.classList.remove('streaming');
    } else {
      bubble.innerHTML = escapeHtml('网络请求失败，请检查服务是否正常运行');
      bubble.classList.remove('streaming');
      bubble.style.color = '#e74c3c';
    }
  } finally {
    finishStreaming(bubble, fullAnswer);
  }
}

/**
 * 流式对话结束后的收尾清理
 * @param {HTMLElement} bubble - 回答气泡 DOM 元素
 * @param {string} fullAnswer - 完整的回答文本
 */
function finishStreaming(bubble, fullAnswer) {
  isStreaming = false;
  abortController = null;
  sendBtn.disabled = false;
  sendBtn.style.display = 'flex';
  stopBtn.style.display = 'none';
  userInput.focus();
  if (!fullAnswer && bubble.classList.contains('streaming')) {
    bubble.innerHTML = escapeHtml('抱歉，未能获取到回答，请重试');
    bubble.classList.remove('streaming');
  }
}

/**
 * 终止当前正在进行的 SSE 流式请求
 */
function stopGeneration() {
  if (abortController) {
    abortController.abort();
  }
}

/* ========== 新对话 ========== */

/**
 * 清空聊天记录，恢复欢迎页面并向服务端申请新会话
 */
async function newChat() {
  if (isStreaming) {
    stopGeneration();
  }
  chatContainer.innerHTML = '';
  const welcomeDiv = document.createElement('div');
  welcomeDiv.className = 'welcome';
  welcomeDiv.id = 'welcome';
  welcomeDiv.innerHTML =
    '<div class="welcome-greeting">欢迎使用智能知识库系统，有什么我能够帮到你的吗？</div>' +
    '<div class="suggestions" id="suggestions"></div>';
  chatContainer.appendChild(welcomeDiv);
  renderSuggestions();
  clearImagePreviews();
  if (getToken()) {
    await createNewSession();
  }
  userInput.focus();
}

/* ========== 图片上传 ========== */

/**
 * 将选择的图片文件读取为 Base64 data URL 并加入待发送列表。
 */
function addImageFiles(files) {
  Array.from(files).forEach(function(file) {
    if (selectedImages.length >= MAX_IMAGES) {
      showToast('最多上传 ' + MAX_IMAGES + ' 张图片');
      return;
    }
    if (!file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = function(e) {
      selectedImages.push(e.target.result);
      renderImagePreviews();
    };
    reader.readAsDataURL(file);
  });
}

/**
 * 渲染待发送图片预览，每张图片带移除按钮。
 */
function renderImagePreviews() {
  imagePreviewArea.innerHTML = '';
  selectedImages.forEach(function(dataUrl, index) {
    const item = document.createElement('div');
    item.className = 'image-preview-item';

    const img = document.createElement('img');
    img.src = dataUrl;
    img.alt = '图片 ' + (index + 1);

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'image-remove-btn';
    removeBtn.textContent = '×';
    removeBtn.title = '移除图片';
    removeBtn.addEventListener('click', function() {
      selectedImages.splice(index, 1);
      renderImagePreviews();
    });

    item.appendChild(img);
    item.appendChild(removeBtn);
    imagePreviewArea.appendChild(item);
  });
}

function clearImagePreviews() {
  selectedImages = [];
  imagePreviewArea.innerHTML = '';
}

/**
 * 拦截输入框粘贴事件，如果剪贴板里有图片则加入待发送列表。
 */
function handleImagePaste(e) {
  const items = e.clipboardData ? e.clipboardData.items : [];
  const imageFiles = [];
  for (const item of items) {
    if (item.type && item.type.startsWith('image/')) {
      const file = item.getAsFile();
      if (file) imageFiles.push(file);
    }
  }
  if (imageFiles.length > 0) {
    e.preventDefault();
    addImageFiles(imageFiles);
  }
}

/* ========== 事件绑定 ========== */
userInput.addEventListener('input', function() {
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
});

userInput.addEventListener('paste', handleImagePaste);

userInput.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);
stopBtn.addEventListener('click', stopGeneration);
imagePickerBtn.addEventListener('click', function() {
  imageInput.click();
});
imageInput.addEventListener('change', function() {
  addImageFiles(imageInput.files);
  imageInput.value = '';
});

document.getElementById('newChatBtn').addEventListener('click', newChat);
loginForm.addEventListener('submit', handleLogin);
document.getElementById('logoutBtn').addEventListener('click', logout);

/* ========== 初始化 ========== */
renderSuggestions();
initModeDropdown();
verifyLogin();
userInput.focus();
