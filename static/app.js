const historyListEl = document.getElementById("historyList");
const appShellEl = document.querySelector(".app-shell");
const chatTitleEl = document.getElementById("chatTitle");
const newChatButtonEl = document.getElementById("newChatButton");
const messagesEl = document.getElementById("messages");
const chatFormEl = document.getElementById("chatForm");
const userInputEl = document.getElementById("userInput");
const followUpSectionEl = document.getElementById("followUpSection");
const followUpChipsEl = document.getElementById("followUpChips");
const sendButtonEl = document.getElementById("sendButton");
const stopButtonEl = document.getElementById("stopButton");
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
let currentStreamAbortController = null;
let isGenerating = false;
let stoppedByUser = false;
let starterSummaryRequestVersion = 0;
const STARTER_SUMMARY_MAX_CONCURRENCY = 2;
const starterSummaryAbortControllers = new Set();
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

function stopCurrentGeneration() {
  if (!isGenerating || !currentStreamAbortController) {
    return;
  }
  stoppedByUser = true;
  currentStreamAbortController.abort();
}

function isAbortError(error) {
  if (!error) {
    return false;
  }
  if (error.name === "AbortError") {
    return true;
  }
  return String(error).toLowerCase().includes("aborted");
}

function setGeneratingState(isGeneratingFlag) {
  isGenerating = Boolean(isGeneratingFlag);
  if (isGenerating) {
    userInputEl.setAttribute("aria-busy", "true");
    sendButtonEl.disabled = true;
    sendButtonEl.textContent = "생성 중";
    stopButtonEl.classList.remove("hidden");
    stopButtonEl.disabled = false;
    return;
  }
  userInputEl.setAttribute("aria-busy", "false");
  sendButtonEl.disabled = false;
  sendButtonEl.textContent = "전송";
  stopButtonEl.disabled = true;
  stopButtonEl.classList.add("hidden");
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
  starterDocs.forEach((doc, index) => {
    if (!doc.visible) {
      return;
    }
    const card = document.createElement("button");
    card.type = "button";
    card.className = "starter-card";
    card.dataset.starterIndex = String(index);
    card.disabled = !doc.summary_ready;
    const title = document.createElement("h4");
    title.textContent = doc.title || "제목 없음";
    const summary = document.createElement("p");
    const summaryReady = Boolean(doc.summary_ready);
    summary.textContent = summaryReady ? (doc.summary || "요약 정보가 없습니다.") : (doc.summary || "");
    if (!summaryReady) {
      summary.classList.add("streaming");
    }
    card.append(title, summary);
    card.addEventListener("click", () => handleStarterClick(doc));
    container.appendChild(card);
  });
}

function getStarterCardsContainer() {
  return messagesEl.querySelector("#starterCards");
}

function ensureStarterCard(index) {
  const container = getStarterCardsContainer();
  if (!container) {
    return null;
  }
  const existing = container.querySelector(`.starter-card[data-starter-index="${index}"]`);
  if (existing) {
    return existing;
  }
  const doc = starterDocs[index];
  if (!doc) {
    return null;
  }
  const card = document.createElement("button");
  card.type = "button";
  card.className = "starter-card";
  card.dataset.starterIndex = String(index);
  card.disabled = !doc.summary_ready;
  const title = document.createElement("h4");
  title.textContent = doc.title || "제목 없음";
  const summary = document.createElement("p");
  summary.textContent = doc.summary || "";
  if (!doc.summary_ready) {
    summary.classList.add("streaming");
  }
  card.append(title, summary);
  card.addEventListener("click", () => handleStarterClick(doc));
  container.appendChild(card);
  return card;
}

function removeStarterCard(index) {
  const card = messagesEl.querySelector(`.starter-card[data-starter-index="${index}"]`);
  if (card) {
    card.remove();
  }
}

function setStarterCardSummary(index, text, ready, useFadeIn = true) {
  const card = ensureStarterCard(index);
  if (!card) {
    return;
  }
  const summaryEl = card.querySelector("p");
  summaryEl.textContent = text;
  summaryEl.classList.remove("streaming", "fade-in");
  card.disabled = !ready;
  if (useFadeIn) {
    void summaryEl.offsetWidth;
    summaryEl.classList.add("fade-in");
  }
}

function appendStarterToken(index, token) {
  const card = ensureStarterCard(index);
  if (!card) {
    return;
  }
  const summaryEl = card.querySelector("p");
  summaryEl.classList.remove("fade-in");
  summaryEl.classList.add("streaming");
  card.disabled = true;
  const chunk = document.createElement("span");
  chunk.className = "starter-token";
  chunk.textContent = token;
  summaryEl.appendChild(chunk);
}

function abortStarterSummaryStreams() {
  starterSummaryAbortControllers.forEach((controller) => {
    try {
      controller.abort();
    } catch (error) {
      // Ignore abort failures.
    }
  });
  starterSummaryAbortControllers.clear();
}

async function resolveStarterSummaryStream(doc, index, requestVersion) {
  let completed = false;
  const streamAbortController = new AbortController();
  starterSummaryAbortControllers.add(streamAbortController);
  try {
    const response = await fetch("/api/starter-docs/summary/stream", {
      method: "POST",
      signal: streamAbortController.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: doc.id,
        title: doc.title,
        summary: doc.seed_summary || doc.summary || "",
        raw_summary: doc.raw_summary || "",
        summary_source: doc.summary_source || "",
      }),
    });

    if (!response.ok) {
      throw new Error(`starter summary ${response.status}`);
    }

    await consumeSse(response, (eventName, payload) => {
      if (requestVersion !== starterSummaryRequestVersion) {
        return;
      }

      if (eventName === "start") {
        starterDocs[index].visible = true;
        starterDocs[index].summary = "";
        starterDocs[index].summary_ready = false;
        ensureStarterCard(index);
        return;
      }

      if (eventName === "token") {
        const token = String(payload?.token || "");
        if (!token) {
          return;
        }
        starterDocs[index].visible = true;
        starterDocs[index].summary = `${starterDocs[index].summary || ""}${token}`;
        appendStarterToken(index, token);
        return;
      }

      if (eventName === "retry") {
        starterDocs[index].visible = false;
        starterDocs[index].summary = "";
        starterDocs[index].summary_ready = false;
        removeStarterCard(index);
        return;
      }

      if (eventName === "complete") {
        const summaryText = String(payload?.summary || "").trim();
        starterDocs[index].visible = true;
        starterDocs[index].summary = summaryText || "요약 생성 실패";
        starterDocs[index].summary_ready = Boolean(summaryText);
        setStarterCardSummary(index, starterDocs[index].summary, starterDocs[index].summary_ready);
        completed = true;
        return;
      }

      if (eventName === "error") {
        starterDocs[index].visible = true;
        starterDocs[index].summary = "요약 생성 실패";
        starterDocs[index].summary_ready = false;
        setStarterCardSummary(index, "요약 생성 실패", false);
        completed = true;
      }
    });
  } catch (error) {
    if (requestVersion !== starterSummaryRequestVersion) {
      return;
    }
    if (isAbortError(error)) {
      return;
    }
    starterDocs[index].visible = true;
    starterDocs[index].summary = "요약 생성 실패";
    starterDocs[index].summary_ready = false;
    setStarterCardSummary(index, "요약 생성 실패", false);
    completed = true;
  } finally {
    starterSummaryAbortControllers.delete(streamAbortController);
  }

  if (requestVersion === starterSummaryRequestVersion && !completed) {
    starterDocs[index].visible = true;
    starterDocs[index].summary = "요약 생성 실패";
    starterDocs[index].summary_ready = false;
    setStarterCardSummary(index, "요약 생성 실패", false);
  }
}

async function hydrateStarterSummaries(requestVersion) {
  starterDocs.forEach((doc, index) => {
    doc.visible = false;
    doc.summary = "";
    doc.summary_ready = false;
  });

  const total = starterDocs.length;
  if (total === 0) {
    return;
  }

  let nextIndex = 0;
  const workerCount = Math.min(STARTER_SUMMARY_MAX_CONCURRENCY, total);
  const workers = Array.from({ length: workerCount }, async () => {
    while (requestVersion === starterSummaryRequestVersion) {
      const index = nextIndex;
      nextIndex += 1;
      if (index >= total) {
        return;
      }
      const doc = starterDocs[index];
      await resolveStarterSummaryStream(doc, index, requestVersion);
    }
  });

  await Promise.all(workers);
}

function restartStarterSummaries() {
  abortStarterSummaryStreams();
  starterDocs = starterDocs.map((doc) => ({
    ...doc,
    summary: "",
    summary_ready: false,
    visible: false,
  }));
  const requestVersion = ++starterSummaryRequestVersion;
  void hydrateStarterSummaries(requestVersion);
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
  starterDocs = (payload.documents || []).map((doc) => ({
    ...doc,
    seed_summary: doc.summary || "",
    summary: "",
    summary_ready: false,
    visible: false,
  }));
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
  stopCurrentGeneration();
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
  if (isGenerating) {
    return;
  }
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
  const requestMessagesRef = activeMessages;
  activeFollowUps = [];
  renderMessages();

  userInputEl.value = "";
  const progressBubble = appendProgressBubble("질의 분석 중");
  const abortController = new AbortController();
  currentStreamAbortController = abortController;
  stoppedByUser = false;
  setGeneratingState(true);

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      signal: abortController.signal,
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
    if (activeMessages !== requestMessagesRef) {
      return;
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
    if (activeMessages !== requestMessagesRef) {
      return;
    }
    if (stoppedByUser || isAbortError(error)) {
      const stoppedText = "답변 생성을 중지했습니다.";
      finalizeProgressBubble(progressBubble, stoppedText);
      activeMessages.push({
        role: "assistant",
        text: stoppedText,
        reasoning: null,
        created_at: new Date().toISOString(),
      });
      activeFollowUps = [];
      renderMessages();
      return;
    }
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
    currentStreamAbortController = null;
    stoppedByUser = false;
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
  const prompt =
    String(doc.summary || "").trim() ||
    String(doc.raw_summary || "").trim() ||
    String(doc.title || "").trim();
  await sendChat(prompt);
}

function createConversation() {
  stopCurrentGeneration();
  activeConversationId = null;
  activeMessages = [];
  activeFollowUps = [];
  showStarters = true;
  closePreviewPanel();
  renderHistoryList();
  renderMessages();
  restartStarterSummaries();
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

stopButtonEl.addEventListener("click", () => {
  stopCurrentGeneration();
});

async function init() {
  await loadStarterDocs();
  await refreshConversations();
  createConversation();
  setGeneratingState(false);
}

init();
