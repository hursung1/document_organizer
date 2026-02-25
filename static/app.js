const historyListEl = document.getElementById("historyList");
const appShellEl = document.querySelector(".app-shell");
const chatTitleEl = document.getElementById("chatTitle");
const newChatButtonEl = document.getElementById("newChatButton");
const messagesEl = document.getElementById("messages");
const chatFormEl = document.getElementById("chatForm");
const userInputEl = document.getElementById("userInput");
const followUpSectionEl = document.getElementById("followUpSection");
const followUpChipsEl = document.getElementById("followUpChips");
const sendButtonEl = chatFormEl.querySelector('button[type="submit"]');
const deleteConfirmModalEl = document.getElementById("deleteConfirmModal");
const confirmDeleteYesEl = document.getElementById("confirmDeleteYes");
const confirmDeleteNoEl = document.getElementById("confirmDeleteNo");
const previewPanelEl = document.getElementById("previewPanel");
const previewTitleEl = document.getElementById("previewTitle");
const previewUrlEl = document.getElementById("previewUrl");
const previewFrameEl = document.getElementById("previewFrame");
const previewCloseButtonEl = document.getElementById("previewCloseButton");
const previewOpenNewTabEl = document.getElementById("previewOpenNewTab");

let starterDocs = [];
let conversations = [];
let activeConversationId = null;
let activeMessages = [];
let activeFollowUps = [];
let showStarters = true;
let pendingDeleteConversationId = null;
const STAGE_LABELS = {
  analyze_query: "질의 분석 중",
  retrieve_docs: "문서 검색 중",
  enrich_with_arxiv_pdf: "arXiv 원문 보강 중",
  generate_answer: "답변 생성 중",
};

function formatTimeFromIso(isoText) {
  if (!isoText) {
    return "";
  }
  const date = new Date(isoText);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  const year = String(date.getFullYear());
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  const hour = String(date.getHours()).padStart(2, "0");
  const min = String(date.getMinutes()).padStart(2, "0");
  return `${year}년 ${month}월 ${day}일 ${hour}:${min}`;
}

function currentConversationTitle() {
  if (!activeConversationId) {
    return "새 채팅";
  }
  const active = conversations.find((item) => item.conversation_id === activeConversationId);
  return active?.title || "대화";
}

function normalizePreviewUrl(rawUrl) {
  const value = String(rawUrl || "").trim();
  if (!value) {
    return null;
  }
  try {
    const parsed = new URL(value);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return null;
    }
    return parsed.toString();
  } catch (error) {
    return null;
  }
}

function openPreviewPanel(url, label = "") {
  const normalized = normalizePreviewUrl(url);
  if (!normalized || !previewPanelEl || !appShellEl) {
    return;
  }
  const title = String(label || "문서 미리보기").trim() || "문서 미리보기";
  previewTitleEl.textContent = title;
  previewUrlEl.textContent = normalized;
  previewFrameEl.src = normalized;
  previewOpenNewTabEl.href = normalized;
  appShellEl.classList.add("preview-open");
}

function closePreviewPanel() {
  if (!appShellEl || !previewFrameEl) {
    return;
  }
  appShellEl.classList.remove("preview-open");
  previewFrameEl.src = "about:blank";
  previewOpenNewTabEl.href = "#";
  previewTitleEl.textContent = "문서 미리보기";
  previewUrlEl.textContent = "";
}

function appendRichText(container, text) {
  container.innerHTML = "";
  const raw = String(text || "");
  const tokenPattern = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)|(https?:\/\/[^\s]+)/g;
  let cursor = 0;
  let match = tokenPattern.exec(raw);

  while (match) {
    const matchIndex = match.index;
    if (matchIndex > cursor) {
      container.appendChild(document.createTextNode(raw.slice(cursor, matchIndex)));
    }

    const markdownLabel = match[1];
    const markdownUrl = match[2];
    const plainUrl = match[3];
    const candidateUrl = markdownUrl || plainUrl || "";
    const normalizedUrl = normalizePreviewUrl(candidateUrl);

    if (normalizedUrl) {
      const anchor = document.createElement("a");
      anchor.className = "doc-link";
      anchor.href = normalizedUrl;
      anchor.target = "_blank";
      anchor.rel = "noopener noreferrer";
      anchor.textContent = markdownLabel || normalizedUrl;
      anchor.addEventListener("click", (event) => {
        event.preventDefault();
        openPreviewPanel(normalizedUrl, markdownLabel || normalizedUrl);
      });
      container.appendChild(anchor);
    } else {
      container.appendChild(document.createTextNode(match[0]));
    }

    cursor = tokenPattern.lastIndex;
    match = tokenPattern.exec(raw);
  }

  if (cursor < raw.length) {
    container.appendChild(document.createTextNode(raw.slice(cursor)));
  }
}

function createReasoningToggle(reasoning) {
  if (!reasoning || !String(reasoning).trim()) {
    return null;
  }
  const reasoningToggle = document.createElement("details");
  reasoningToggle.className = "reasoning-toggle";

  const summary = document.createElement("summary");
  summary.textContent = "Thinking 보기";

  const content = document.createElement("pre");
  content.className = "reasoning-content";
  content.textContent = String(reasoning).trim();

  reasoningToggle.appendChild(summary);
  reasoningToggle.appendChild(content);
  return reasoningToggle;
}

function appendMessageNode(role, text, reasoning = null) {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;

  if (role === "assistant") {
    const reasoningToggle = createReasoningToggle(reasoning);
    if (reasoningToggle) {
      wrapper.appendChild(reasoningToggle);
    }
  }

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  appendRichText(bubble, text);

  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrapper;
}

function appendProgressBubble(initialStage = "질의 분석 중") {
  const wrapper = document.createElement("div");
  wrapper.className = "msg assistant";

  const bubble = document.createElement("div");
  bubble.className = "bubble loading";
  bubble.innerHTML = `
    <span class="spinner" aria-hidden="true"></span>
    <span class="progress-label">${initialStage}</span>
    <span class="dot-wave" aria-hidden="true"><span>.</span><span>.</span><span>.</span></span>
  `;

  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  return {
    wrapper,
    bubble,
    label: bubble.querySelector(".progress-label"),
  };
}

function setProgressStage(progressBubble, stageText) {
  if (!progressBubble?.label) {
    return;
  }
  progressBubble.label.textContent = stageText;
}

function finalizeProgressBubble(progressBubble, finalText, reasoning = null) {
  if (!progressBubble?.bubble) {
    return;
  }
  const existingToggle = progressBubble.wrapper.querySelector(".reasoning-toggle");
  if (existingToggle) {
    existingToggle.remove();
  }
  const reasoningToggle = createReasoningToggle(reasoning);
  if (reasoningToggle) {
    progressBubble.wrapper.insertBefore(reasoningToggle, progressBubble.bubble);
  }
  progressBubble.bubble.classList.remove("loading");
  appendRichText(progressBubble.bubble, finalText);
}

function setGeneratingState(isGenerating) {
  if (isGenerating) {
    userInputEl.setAttribute("aria-busy", "true");
    sendButtonEl.disabled = true;
    sendButtonEl.textContent = "생성 중";
    return;
  }
  userInputEl.setAttribute("aria-busy", "false");
  sendButtonEl.disabled = false;
  sendButtonEl.textContent = "전송";
}

function renderFollowUps(questions) {
  followUpChipsEl.innerHTML = "";
  (questions || []).forEach((q) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chip";
    button.textContent = q;
    button.addEventListener("click", () => {
      userInputEl.value = q;
      userInputEl.focus();
    });
    followUpChipsEl.appendChild(button);
  });

  if ((questions || []).length > 0) {
    followUpSectionEl.classList.remove("hidden");
  } else {
    followUpSectionEl.classList.add("hidden");
  }
}

function renderHistoryList() {
  historyListEl.innerHTML = "";
  const sorted = [...conversations].sort(
    (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
  );

  sorted.forEach((conversation) => {
    const item = document.createElement("div");
    item.className = `history-item ${
      conversation.conversation_id === activeConversationId ? "active" : ""
    }`;
    item.setAttribute("role", "button");
    item.setAttribute("tabindex", "0");
    item.innerHTML = `
      <div class="history-main">
        <div class="label">${conversation.title || "새 대화"}</div>
        <div class="meta">${formatTimeFromIso(conversation.updated_at)}</div>
      </div>
      <button type="button" class="history-delete">삭제</button>
    `;
    const deleteButton = item.querySelector(".history-delete");
    deleteButton.addEventListener("click", (event) => {
      event.stopPropagation();
      openDeleteModal(conversation.conversation_id);
    });

    item.addEventListener("click", (event) => {
      if (event.target.closest(".history-delete")) {
        return;
      }
      openConversation(conversation.conversation_id);
    });
    item.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openConversation(conversation.conversation_id);
      }
    });

    historyListEl.appendChild(item);
  });
}

function openDeleteModal(conversationId) {
  pendingDeleteConversationId = conversationId;
  deleteConfirmModalEl.classList.remove("hidden");
}

function closeDeleteModal() {
  pendingDeleteConversationId = null;
  deleteConfirmModalEl.classList.add("hidden");
}

async function confirmDeleteConversation() {
  const conversationId = pendingDeleteConversationId;
  if (!conversationId) {
    closeDeleteModal();
    return;
  }

  try {
    const response = await fetch(`/api/conversations/${conversationId}`, {
      method: "DELETE",
    });
    if (!response.ok && response.status !== 404) {
      let detail = "삭제 실패";
      try {
        const payload = await response.json();
        if (payload?.detail) {
          detail = payload.detail;
        }
      } catch (error) {
        // Keep fallback message.
      }
      throw new Error(detail);
    }

    await refreshConversations();
    if (activeConversationId === conversationId) {
      createConversation();
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    window.alert(`대화 삭제 중 오류가 발생했습니다: ${message}`);
  } finally {
    closeDeleteModal();
  }
}

function renderStarterButtons(container) {
  container.innerHTML = "";
  starterDocs.forEach((doc) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "starter-card";
    card.innerHTML = `<h4>${doc.title}</h4><p>${doc.summary}</p>`;
    card.addEventListener("click", () => handleStarterClick(doc));
    container.appendChild(card);
  });
}

function renderMessages() {
  messagesEl.innerHTML = "";
  chatTitleEl.textContent = currentConversationTitle();

  if (showStarters && activeMessages.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.innerHTML = `
      <h2>질문을 입력하거나 추천 문서를 선택하세요</h2>
      <div class="starter-cards" id="starterCards"></div>
    `;
    messagesEl.appendChild(empty);
    renderStarterButtons(empty.querySelector("#starterCards"));
    renderFollowUps([]);
    return;
  }

  activeMessages.forEach((message) => {
    appendMessageNode(message.role, message.text, message.reasoning || null);
  });
  renderFollowUps(activeFollowUps);
}

function hideStarterCardsImmediately() {
  const emptyState = messagesEl.querySelector(".empty-state");
  if (emptyState) {
    emptyState.remove();
  }
}

async function loadStarterDocs() {
  const response = await fetch("/api/starter-docs");
  const payload = await response.json();
  starterDocs = payload.documents || [];
}

async function refreshConversations() {
  const response = await fetch("/api/conversations");
  if (!response.ok) {
    conversations = [];
    renderHistoryList();
    return;
  }
  const payload = await response.json();
  conversations = payload.conversations || [];
  renderHistoryList();
}

async function openConversation(conversationId) {
  activeConversationId = conversationId;
  showStarters = false;
  activeFollowUps = [];
  closePreviewPanel();

  const response = await fetch(`/api/conversations/${conversationId}/messages`);
  if (!response.ok) {
    activeMessages = [];
    renderHistoryList();
    renderMessages();
    return;
  }

  const payload = await response.json();
  activeMessages = payload.messages || [];
  renderHistoryList();
  renderMessages();
}

async function sendChat(messageText) {
  const message = (messageText || "").trim();
  if (!message) {
    return;
  }

  showStarters = false;
  hideStarterCardsImmediately();

  const userMessage = {
    role: "user",
    text: message,
    created_at: new Date().toISOString(),
  };
  activeMessages.push(userMessage);
  activeFollowUps = [];
  renderMessages();

  userInputEl.value = "";
  const progressBubble = appendProgressBubble("질의 분석 중");
  setGeneratingState(true);

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        conversation_id: activeConversationId,
      }),
    });
    if (!response.ok) {
      let detail = "답변 실패";
      try {
        const errorPayload = await response.json();
        if (errorPayload && typeof errorPayload.detail === "string" && errorPayload.detail.trim()) {
          detail = errorPayload.detail;
        }
      } catch (parseError) {
        // Keep fallback detail when response body is not JSON.
      }
      throw new Error(detail);
    }

    let finalPayload = null;
    let streamError = null;
    await consumeSse(response, (eventName, data) => {
      if (eventName === "stage") {
        const stage = (data?.stage || "").trim();
        const label = data?.label || STAGE_LABELS[stage] || "답변 생성 중";
        setProgressStage(progressBubble, label);
        return;
      }
      if (eventName === "final") {
        finalPayload = data;
        return;
      }
      if (eventName === "error") {
        streamError = data?.detail || "스트리밍 중 오류가 발생했습니다.";
      }
    });

    if (streamError) {
      throw new Error(streamError);
    }
    if (!finalPayload) {
      throw new Error("최종 답변 이벤트를 받지 못했습니다.");
    }

    activeConversationId = finalPayload.conversation_id;
    const finalAnswer = finalPayload.answer || "답변이 없습니다.";
    finalizeProgressBubble(progressBubble, finalAnswer, finalPayload.reasoning || null);
    activeMessages.push({
      role: "assistant",
      text: finalAnswer,
      reasoning: finalPayload.reasoning || null,
      created_at: new Date().toISOString(),
    });
    activeFollowUps = finalPayload.suggested_questions || [];

    await refreshConversations();
    renderMessages();
  } catch (error) {
    const errorText = error instanceof Error ? error.message : String(error);
    const finalError = `답변 생성 중 오류가 발생했습니다: ${errorText}`;
    finalizeProgressBubble(progressBubble, finalError);
    activeMessages.push({
      role: "assistant",
      text: finalError,
      reasoning: null,
      created_at: new Date().toISOString(),
    });
    activeFollowUps = [];
    renderMessages();
  } finally {
    setGeneratingState(false);
  }
}

async function consumeSse(response, onEvent) {
  if (!response.body) {
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    buffer = buffer.replace(/\r\n/g, "\n");

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const chunk = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);

      if (chunk) {
        let eventName = "message";
        const dataLines = [];
        chunk.split("\n").forEach((line) => {
          if (line.startsWith("event:")) {
            eventName = line.slice(6).trim();
            return;
          }
          if (line.startsWith("data:")) {
            dataLines.push(line.slice(5).trimStart());
          }
        });

        let payload = {};
        const dataText = dataLines.join("\n");
        if (dataText) {
          try {
            payload = JSON.parse(dataText);
          } catch (error) {
            payload = {};
          }
        }
        onEvent(eventName, payload);
      }

      boundary = buffer.indexOf("\n\n");
    }
  }
}

async function handleStarterClick(doc) {
  await sendChat(doc.summary);
}

function createConversation() {
  activeConversationId = null;
  activeMessages = [];
  activeFollowUps = [];
  showStarters = true;
  closePreviewPanel();
  renderHistoryList();
  renderMessages();
}

chatFormEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  await sendChat(userInputEl.value);
});

userInputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatFormEl.requestSubmit();
  }
});

newChatButtonEl.addEventListener("click", () => {
  createConversation();
});

confirmDeleteYesEl.addEventListener("click", () => {
  confirmDeleteConversation();
});

confirmDeleteNoEl.addEventListener("click", () => {
  closeDeleteModal();
});

previewCloseButtonEl.addEventListener("click", () => {
  closePreviewPanel();
});

async function init() {
  await loadStarterDocs();
  await refreshConversations();
  createConversation();
  setGeneratingState(false);
}

init();
