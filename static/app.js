const historyListEl = document.getElementById("historyList");
const chatTitleEl = document.getElementById("chatTitle");
const newChatButtonEl = document.getElementById("newChatButton");
const messagesEl = document.getElementById("messages");
const chatFormEl = document.getElementById("chatForm");
const userInputEl = document.getElementById("userInput");
const followUpSectionEl = document.getElementById("followUpSection");
const followUpChipsEl = document.getElementById("followUpChips");

let starterDocs = [];
let conversations = [];
let activeConversationId = null;
let activeMessages = [];
let activeFollowUps = [];
let showStarters = true;

function formatTimeFromIso(isoText) {
  if (!isoText) {
    return "";
  }
  const date = new Date(isoText);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  const hour = String(date.getHours()).padStart(2, "0");
  const min = String(date.getMinutes()).padStart(2, "0");
  return `${hour}:${min}`;
}

function currentConversationTitle() {
  if (!activeConversationId) {
    return "새 채팅";
  }
  const active = conversations.find((item) => item.conversation_id === activeConversationId);
  return active?.title || "대화";
}

function appendMessageNode(role, text) {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrapper;
}

function appendLoadingMessage(text = "답변 생성 중...") {
  return appendMessageNode("assistant", text);
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
    const button = document.createElement("button");
    button.type = "button";
    button.className = `history-item ${
      conversation.conversation_id === activeConversationId ? "active" : ""
    }`;
    button.innerHTML = `
      <div class="label">${conversation.title || "새 대화"}</div>
      <div class="meta">${formatTimeFromIso(conversation.updated_at)}</div>
    `;
    button.addEventListener("click", () => openConversation(conversation.conversation_id));
    historyListEl.appendChild(button);
  });
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
    appendMessageNode(message.role, message.text);
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
  const loadingNode = appendLoadingMessage();

  try {
    const response = await fetch("/api/chat", {
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

    const payload = await response.json();
    loadingNode.remove();
    activeConversationId = payload.conversation_id;
    activeMessages.push({
      role: "assistant",
      text: payload.answer || "답변이 없습니다.",
      created_at: new Date().toISOString(),
    });
    activeFollowUps = payload.suggested_questions || [];

    await refreshConversations();
    renderMessages();
  } catch (error) {
    loadingNode.remove();
    const errorText = error instanceof Error ? error.message : String(error);
    activeMessages.push({
      role: "assistant",
      text: `답변 생성 중 오류가 발생했습니다: ${errorText}`,
      created_at: new Date().toISOString(),
    });
    activeFollowUps = [];
    renderMessages();
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

async function init() {
  await loadStarterDocs();
  await refreshConversations();
  createConversation();
}

init();
