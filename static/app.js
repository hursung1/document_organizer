const starterCardsEl = document.getElementById("starterCards");
const messagesEl = document.getElementById("messages");
const chatFormEl = document.getElementById("chatForm");
const userInputEl = document.getElementById("userInput");
const followUpSectionEl = document.getElementById("followUpSection");
const followUpChipsEl = document.getElementById("followUpChips");

let activeDocumentId = null;

function appendMessage(role, text) {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendLoadingMessage(text = "답변 생성 중...") {
  const wrapper = document.createElement("div");
  wrapper.className = "msg assistant";
  wrapper.dataset.loading = "true";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrapper;
}

function resetConversation() {
  messagesEl.innerHTML = "";
  followUpSectionEl.classList.add("hidden");
  followUpChipsEl.innerHTML = "";
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

async function loadStarterCards() {
  const response = await fetch("/api/starter-docs");
  const payload = await response.json();
  const docs = payload.documents || [];

  starterCardsEl.innerHTML = "";
  docs.forEach((doc) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "starter-card";
    card.innerHTML = `<h4>${doc.title}</h4><p>${doc.summary}</p>`;
    card.addEventListener("click", () => startConversation(doc.id));
    starterCardsEl.appendChild(card);
  });
}

async function startConversation(documentId) {
  activeDocumentId = documentId;
  resetConversation();

  try {
    const response = await fetch("/api/chat/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_id: documentId }),
    });
    if (!response.ok) {
      throw new Error("문서 대화를 시작하지 못했습니다.");
    }
    const payload = await response.json();
    appendMessage("assistant", payload.answer || "답변이 없습니다.");
    renderFollowUps(payload.suggested_questions || []);
  } catch (error) {
    appendMessage("assistant", "문서 대화 시작 중 오류가 발생했습니다.");
    renderFollowUps([]);
  }
}

chatFormEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = userInputEl.value.trim();
  if (!message) {
    return;
  }

  appendMessage("user", message);
  userInputEl.value = "";
  const loadingNode = appendLoadingMessage();

  try {
    const useDocumentChat = Boolean(activeDocumentId);
    const endpoint = useDocumentChat ? "/api/chat" : "/api/rag/chat";
    const body = useDocumentChat
      ? JSON.stringify({ document_id: activeDocumentId, message })
      : JSON.stringify({ message });

    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
    if (!response.ok) {
      throw new Error("답변 실패");
    }
    const payload = await response.json();
    loadingNode.remove();
    appendMessage("assistant", payload.answer || "답변이 없습니다.");
    renderFollowUps(payload.suggested_questions || []);
  } catch (error) {
    loadingNode.remove();
    appendMessage("assistant", "답변 생성 중 오류가 발생했습니다.");
    renderFollowUps([]);
  }
});

userInputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatFormEl.requestSubmit();
  }
});

loadStarterCards();
