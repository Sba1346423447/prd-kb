/**
 * 企业技术支持智研知识库（PRD-KB）—— 前端交互脚本
 *
 * 核心能力：SSE 流式对话、Markdown 渲染、思考过程可视化。
 * 依赖：marked.js（Markdown 解析库，通过 CDN 加载）。
 */

/* ========== DOM 元素引用 ========== */
const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const stopBtn = document.getElementById('stopBtn');
const toast = document.getElementById('toast');

/* ========== 全局状态 ========== */
let isStreaming = false;
let abortController = null;

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
 * 获取当前会话 ID
 * @returns {string} 会话 ID，未填写时返回 default
 */
function getSessionId() {
  return document.getElementById('sessionId').value.trim() || 'default';
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
function addMessage(role, content, isStreamingMsg) {
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
    msgDiv.innerHTML =
      '<div class="avatar">U</div>' +
      '<div class="bubble">' + escapeHtml(content) + '</div>';
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

  isStreaming = true;
  sendBtn.disabled = true;
  sendBtn.style.display = 'none';
  stopBtn.style.display = 'flex';
  userInput.value = '';
  userInput.style.height = 'auto';

  addMessage('user', question);
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
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: question,
        session_id: getSessionId()
      }),
      signal: abortController.signal
    });

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
 * 清空聊天记录，恢复欢迎页面与推荐问题卡片
 */
function newChat() {
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
  userInput.focus();
}

/* ========== 事件绑定 ========== */
userInput.addEventListener('input', function() {
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
});

userInput.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);
stopBtn.addEventListener('click', stopGeneration);

document.getElementById('newChatBtn').addEventListener('click', newChat);

/* ========== 初始化 ========== */
renderSuggestions();
userInput.focus();