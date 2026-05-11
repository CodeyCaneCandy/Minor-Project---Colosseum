/**
 * chatbot.js — Colosseum AI Assistant
 * Floating chat widget backed by /api/chat (Gemini via llm_explainer.py)
 *
 * Reads `window.colonneState` which app.js populates:
 *   window.colonneState = { session, results }
 * Falls back gracefully if it hasn't been set yet.
 */

(function () {
  'use strict';

  // ── DOM refs ──────────────────────────────────────────────────────────────
  const toggle     = document.getElementById('chatbot-toggle');
  const panel      = document.getElementById('chatbot-panel');
  const messages   = document.getElementById('chatbot-messages');
  const input      = document.getElementById('chatbot-input');
  const sendBtn    = document.getElementById('chatbot-send');
  const clearBtn   = document.getElementById('chatbot-clear');
  const badge      = document.getElementById('chatbot-badge');
  const liveDot    = document.getElementById('chatbot-live-dot');
  const statusLbl  = document.getElementById('chatbot-status-label');
  const iconOpen   = document.querySelector('.chatbot-icon-open');
  const iconClose  = document.querySelector('.chatbot-icon-close');

  // ── State ─────────────────────────────────────────────────────────────────
  let isOpen       = false;
  let isThinking   = false;
  let unreadCount  = 0;
  // Conversation history in Gemini format: [{role, text}, ...]
  const history    = [];

  // ── Helpers ───────────────────────────────────────────────────────────────
  function getAppState() {
    // app.js writes this when results arrive — we read it here
    return window.colonneState || { session: null, results: null };
  }

  function openPanel() {
    isOpen = true;
    panel.classList.remove('hidden');
    panel.classList.add('chatbot-panel-open');
    iconOpen.classList.add('hidden');
    iconClose.classList.remove('hidden');
    toggle.classList.add('chatbot-toggle-active');
    unreadCount = 0;
    badge.classList.add('hidden');
    badge.textContent = '0';
    input.focus();
    scrollBottom();
  }

  function closePanel() {
    isOpen = false;
    panel.classList.remove('chatbot-panel-open');
    panel.classList.add('hidden');
    iconOpen.classList.remove('hidden');
    iconClose.classList.add('hidden');
    toggle.classList.remove('chatbot-toggle-active');
  }

  function scrollBottom() {
    requestAnimationFrame(() => {
      messages.scrollTop = messages.scrollHeight;
    });
  }

  function setStatus(text) {
    statusLbl.textContent = text;
  }

  function setBusy(busy) {
    isThinking = busy;
    sendBtn.disabled = busy;
    input.disabled   = busy;
    liveDot.classList.toggle('chatbot-live-dot-active', busy);
    if (busy) setStatus('Thinking…');
    else {
      const state = getAppState();
      setStatus(state.results ? 'Results loaded — ask anything' : 'Ask me about your results');
    }
  }

  // ── Render a message bubble ───────────────────────────────────────────────
  function appendMessage(role, text) {
    const wrapper = document.createElement('div');
    wrapper.className = `chatbot-msg chatbot-msg-${role === 'user' ? 'user' : 'assistant'}`;

    const bubble = document.createElement('div');
    bubble.className = 'chatbot-bubble';
    // Light markdown: **bold**, `code`, newlines
    bubble.innerHTML = formatText(text);

    wrapper.appendChild(bubble);
    messages.appendChild(wrapper);
    scrollBottom();

    if (!isOpen && role !== 'user') {
      unreadCount++;
      badge.textContent = unreadCount;
      badge.classList.remove('hidden');
    }
  }

  function appendThinkingDots() {
    const wrapper = document.createElement('div');
    wrapper.className = 'chatbot-msg chatbot-msg-assistant';
    wrapper.id = 'chatbot-thinking';
    wrapper.innerHTML = '<div class="chatbot-bubble chatbot-thinking"><span></span><span></span><span></span></div>';
    messages.appendChild(wrapper);
    scrollBottom();
  }

  function removeThinkingDots() {
    const el = document.getElementById('chatbot-thinking');
    if (el) el.remove();
  }

  function formatText(text) {
    return text
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br>');
  }

  // ── Send a message ────────────────────────────────────────────────────────
  async function sendMessage() {
    const text = input.value.trim();
    if (!text || isThinking) return;

    input.value = '';
    appendMessage('user', text);
    setBusy(true);
    appendThinkingDots();

    const { session, results } = getAppState();

    // Build payload
    const payload = {
      message: text,
      history: history.map(h => ({ role: h.role, text: h.text })),
      session: session || null,
      results: results || null,
    };

    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await resp.json();
      const reply = data.reply || '(no reply)';

      // Save to history
      history.push({ role: 'user',  text });
      history.push({ role: 'model', text: reply });

      removeThinkingDots();
      appendMessage('model', reply);

    } catch (err) {
      removeThinkingDots();
      appendMessage('model', `❌ Network error: ${err.message}`);
    }

    setBusy(false);
  }

  // ── When results arrive from app.js — auto-brief the user ────────────────
  function onResultsReady(results, session) {
    // Show a badge nudge even if panel is closed
    if (!isOpen) {
      unreadCount++;
      badge.textContent = '!';
      badge.classList.remove('hidden');
    }
    setStatus('Results loaded — ask anything');
  }

  // ── app.js hook: call window.chatbotOnResults(results, session) ──────────
  window.chatbotOnResults = onResultsReady;

  // ── Event listeners ───────────────────────────────────────────────────────
  toggle.addEventListener('click', () => isOpen ? closePanel() : openPanel());

  sendBtn.addEventListener('click', sendMessage);

  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  clearBtn.addEventListener('click', () => {
    history.length = 0;
    messages.innerHTML = '';
    appendMessage('model', 'Chat cleared! Ask me anything about your dataset or results.');
  });

  // Close on Escape
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && isOpen) closePanel();
  });

})();
