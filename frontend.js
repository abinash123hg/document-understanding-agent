const APIBASE = "http://127.0.0.1:8000";
const HISTORYKEY = "documentagenthistorybydocumentv3";

const state = {
  documents: [],
  selectedDocument: null,
  historyByDocument: loadHistory()
};

const $ = (id) => document.getElementById(id);

function loadHistory() {
  try {
    const value = JSON.parse(localStorage.getItem(HISTORYKEY));
    return value && typeof value === "object" && !Array.isArray(value)
      ? value
      : {};
  } catch {
    return {};
  }
}

function saveHistory() {
  localStorage.setItem(HISTORYKEY, JSON.stringify(state.historyByDocument));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setStatus(text, kind = "ready") {
  if (!$("status")) return;
  $("status").textContent = text;
  $("status").className = `status ${kind}`;
}

function showError(error) {
  return error?.message || "Request failed.";
}

function renderDocuments() {
  const box = $("documents");
  if (!box) return;

  if (!state.documents.length) {
    box.innerHTML = `<div class="muted">No document uploaded.</div>`;
    return;
  }

  box.innerHTML = state.documents
    .map((name) => `
      <div class="document-item ${state.selectedDocument === name ? "active" : ""}">
        <button type="button" data-document="${escapeHtml(name)}">
          ${escapeHtml(name)}
        </button>
      </div>
    `)
    .join("");

  box.querySelectorAll("[data-document]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedDocument = button.dataset.document;
      renderDocuments();
      renderHistory();

      if ($("chat")) {
        $("chat").innerHTML = "";
        addBubble(
          "assistant",
          `Selected document: ${state.selectedDocument}. I will use only this document.`
        );
      }

      setStatus("Ready", "ready");
    });
  });
}

function renderHistory() {
  const list = $("historyList");
  const title = $("historyDocument");

  if (!list) return;

  if (title) {
    title.textContent = state.selectedDocument
      ? `History: ${state.selectedDocument}`
      : "Select a document.";
  }

  const items = state.selectedDocument
    ? state.historyByDocument[state.selectedDocument] || []
    : [];

  if (!items.length) {
    list.innerHTML = `<div class="muted">No questions for this document.</div>`;
    return;
  }

  list.innerHTML = items
    .map((item) => `
      <button class="history-item" type="button">
        <span class="history-question">${escapeHtml(item.question)}</span>
        <span class="history-meta">${escapeHtml(item.time)}</span>
      </button>
    `)
    .join("");
}

function addBubble(role, text, sources = []) {
  const chat = $("chat");
  if (!chat) return;

  const empty = chat.querySelector(".empty-state");
  if (empty) empty.remove();

  const sourceHtml = Array.isArray(sources) && sources.length
    ? `<div class="sources"><strong>Source:</strong> ${
        sources.map(escapeHtml).join(", ")
      }</div>`
    : "";

  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  bubble.innerHTML = `
    <span class="bubble-label">${role === "user" ? "You" : "Agent"}</span>
    <div class="bubble-content">${escapeHtml(text)}</div>
    ${sourceHtml}
  `;

  chat.appendChild(bubble);
  chat.scrollTop = chat.scrollHeight;
}

function getPayloadError(payload, fallback) {
  if (!payload) return fallback;
  if (typeof payload.detail === "string") return payload.detail;
  if (typeof payload.message === "string") return payload.message;
  return fallback;
}

async function uploadFile() {
  const file = $("fileInput")?.files?.[0];

  if (!file) {
    if ($("uploadStatus")) $("uploadStatus").textContent = "Choose a file first.";
    return;
  }

  const form = new FormData();
  form.append("file", file);

  setStatus("Uploading", "busy");

  try {
    const response = await fetch(`${APIBASE}/upload`, {
      method: "POST",
      body: form
    });

    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      throw new Error(getPayloadError(payload, `Upload failed (${response.status})`));
    }

    const name = payload.documentname || payload.filename || file.name;

    if (!state.documents.includes(name)) {
      state.documents.push(name);
    }

    state.selectedDocument = name;
    renderDocuments();
    renderHistory();

    if ($("chat")) {
      $("chat").innerHTML = "";
      addBubble(
        "assistant",
        `Ready. I will answer only from ${name}.`
      );
    }

    if ($("uploadStatus")) {
      $("uploadStatus").textContent = `Uploaded ${name}`;
    }

    setStatus("Ready", "ready");
  } catch (error) {
    if ($("uploadStatus")) {
      $("uploadStatus").textContent = `Upload failed: ${showError(error)}`;
    }
    setStatus("Backend error", "error");
  }
}

async function ask(event) {
  event.preventDefault();

  const input = $("question");
  const question = input?.value.trim();

  if (!question || !state.selectedDocument) {
    addBubble("assistant", "Select a document and enter a question.");
    return;
  }

  addBubble("user", question);
  input.value = "";

  if ($("askBtn")) $("askBtn").disabled = true;
  setStatus("Thinking", "busy");

  try {
    const response = await fetch(`${APIBASE}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        question,
        documentname: state.selectedDocument
      })
    });

    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      throw new Error(getPayloadError(payload, `Request failed (${response.status})`));
    }

    const answer = payload.answer || payload.response;

    if (!answer) {
      throw new Error("The backend returned no answer.");
    }

    addBubble("assistant", answer, payload.sources || []);

    if (!state.historyByDocument[state.selectedDocument]) {
      state.historyByDocument[state.selectedDocument] = [];
    }

    state.historyByDocument[state.selectedDocument].unshift({
      question,
      answer,
      time: new Date().toLocaleString()
    });

    state.historyByDocument[state.selectedDocument] =
      state.historyByDocument[state.selectedDocument].slice(0, 50);

    saveHistory();
    renderHistory();
    setStatus("Ready", "ready");
  } catch (error) {
    addBubble("assistant", `I could not answer safely: ${showError(error)}`);
    setStatus("Backend error", "error");
  } finally {
    if ($("askBtn")) $("askBtn").disabled = false;
  }
}

$("uploadBtn")?.addEventListener("click", uploadFile);
$("chatForm")?.addEventListener("submit", ask);

$("clearHistoryBtn")?.addEventListener("click", () => {
  if (!state.selectedDocument) return;
  delete state.historyByDocument[state.selectedDocument];
  saveHistory();
  renderHistory();
});

renderDocuments();
renderHistory();
setStatus("Ready", "ready");
