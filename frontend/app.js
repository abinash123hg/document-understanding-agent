const API_BASE = "http://127.0.0.1:8000";
const HISTORY_KEY = "document_agent_history_by_document_v3";

const state = {
  documents: [],
  selectedDocument: null,
  historyByDocument: loadHistory(),
  theme: localStorage.getItem("document_agent_theme") || "light"
};

const $ = (id) => document.getElementById(id);

function loadHistory() {
  try {
    const value = JSON.parse(localStorage.getItem(HISTORY_KEY) || "{}");
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  } catch {
    return {};
  }
}

function saveHistory() {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(state.historyByDocument));
}

function setTheme(theme) {
  state.theme = theme;
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("document_agent_theme", theme);
  $("themeBtn").textContent = theme === "dark" ? "Light" : "Dark";
}

function setStatus(text, kind = "ready") {
  $("status").textContent = text;
  $("status").className = `status ${kind}`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[char]));
}

function currentHistory() {
  return state.selectedDocument ? (state.historyByDocument[state.selectedDocument] || []) : [];
}

function renderHistory() {
  const title = $("historyDocument");
  const list = $("historyList");

  title.textContent = state.selectedDocument
    ? `History: ${state.selectedDocument}`
    : "Select a document.";

  const items = currentHistory();

  if (!items.length) {
    list.innerHTML = '<div class="muted">No questions for this document.</div>';
    return;
  }

  list.innerHTML = items.map((item, index) => `
    <button class="history-item" data-history-index="${index}" type="button">
      <span class="history-question">${escapeHtml(item.question)}</span>
      <span class="history-meta">${escapeHtml(item.time)}</span>
    </button>
  `).join("");

  list.querySelectorAll("[data-history-index]").forEach((button) => {
    button.addEventListener("click", () => {
      const item = currentHistory()[Number(button.dataset.historyIndex)];
      if (!item) return;
      addBubble("user", item.question);
      addBubble("assistant", item.answer, item.sources);
    });
  });
}

function renderDocuments() {
  const box = $("documents");

  if (!state.documents.length) {
    box.innerHTML = '<div class="muted">No document uploaded.</div>';
    return;
  }

  box.innerHTML = state.documents.map((name) => `
    <div class="document-item ${name === state.selectedDocument ? "active" : ""}">
      <button type="button" data-document="${escapeHtml(name)}">${escapeHtml(name)}</button>
    </div>
  `).join("");

  box.querySelectorAll("[data-document]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedDocument = button.dataset.document;
      renderDocuments();
      renderHistory();
      $("chat").innerHTML = "";
      addBubble("assistant", `Selected document: ${state.selectedDocument}\nI will use only this document.`);
      $("question").disabled = false;
      $("askBtn").disabled = false;
    });
  });
}

function addBubble(role, text, sources = []) {
  const chat = $("chat");
  const empty = chat.querySelector(".empty-state");
  if (empty) empty.remove();

  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  const safeSources = Array.isArray(sources) ? sources.filter(Boolean) : [];

  bubble.innerHTML = `
    <span class="bubble-label">${role === "user" ? "You" : "Agent"}</span>
    <div class="bubble-content">${escapeHtml(text)}</div>
    ${role === "assistant" && safeSources.length ? `
      <div class="sources"><strong>Sources from ${escapeHtml(state.selectedDocument || "selected document")}</strong><br>
      ${safeSources.map((source) => escapeHtml(source)).join("<br>")}</div>
    ` : ""}
  `;

  chat.appendChild(bubble);
  chat.scrollTop = chat.scrollHeight;
}

function errorMessage(payload, fallback) {
  if (!payload) return fallback;
  if (typeof payload === "string") return payload;
  if (typeof payload.detail === "string") return payload.detail;
  if (typeof payload.message === "string") return payload.message;
  if (Array.isArray(payload.detail)) {
    return payload.detail.map((x) => typeof x === "string" ? x : (x.msg || JSON.stringify(x))).join("; ");
  }
  return payload.detail ? JSON.stringify(payload.detail) : fallback;
}

function answerText(payload) {
  if (typeof payload.answer === "string") return payload.answer;
  if (typeof payload.response === "string") return payload.response;
  if (typeof payload.message === "string") return payload.message;
  return "No answer returned.";
}

function sourcesFrom(payload) {
  const sources = payload.sources || payload.source_chunks || payload.citations || [];
  if (!Array.isArray(sources)) return [];
  return sources.map((x) => typeof x === "string" ? x : (x.filename || x.file_name || x.source || "")).filter(Boolean);
}

async function checkBackend() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    const payload = await response.json();
    if (!response.ok || payload.backend !== "ok") throw new Error("Backend health check failed.");
    if (!payload.ollama) return setStatus("Ollama offline", "error");
    if (!payload.model_available) return setStatus(`Model missing: ${payload.model}`, "error");
    setStatus("Ready", "ready");
  } catch (error) {
    setStatus("Backend offline", "error");
    console.warn("Backend health check failed:", error);
  }
}

async function uploadFile() {
  const file = $("fileInput").files[0];

  if (!file) {
    $("uploadStatus").textContent = "Choose one file first.";
    $("uploadStatus").className = "message error";
    return;
  }

  const form = new FormData();
  form.append("file", file);

  setStatus("Uploading", "busy");
  $("uploadStatus").textContent = "Processing document...";
  $("uploadStatus").className = "message";

  try {
    const response = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(errorMessage(payload, `Upload failed (${response.status})`));
    }

    const name = payload.filename || payload.file_name || payload.name || file.name;
    state.documents = [...new Set([...state.documents, name])];
    state.selectedDocument = name;

    if (!state.historyByDocument[name]) state.historyByDocument[name] = [];
    saveHistory();

    renderDocuments();
    renderHistory();

    $("chat").innerHTML = "";
    addBubble("assistant", `Ready. I will answer only from: ${name}`);
    $("question").disabled = false;
    $("askBtn").disabled = false;
    $("uploadStatus").textContent = `Uploaded: ${name}${payload.chunks_processed ? ` · ${payload.chunks_processed} chunks processed` : ""}`;
    $("uploadStatus").className = "message success";
    setStatus("Ready", "ready");
  } catch (error) {
    $("uploadStatus").textContent = `Upload failed: ${error.message}`;
    $("uploadStatus").className = "message error";
    setStatus("Backend error", "error");
  }
}

async function ask(event) {
  event.preventDefault();

  const question = $("question").value.trim();
  if (!question || !state.selectedDocument) return;

  addBubble("user", question);
  $("question").value = "";
  $("askBtn").disabled = true;
  setStatus("Thinking", "busy");

  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        question,
        document_name: state.selectedDocument,
      })
    });

    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(errorMessage(payload, `Request failed (${response.status})`));
    }

    const answer = answerText(payload);
    const sources = sourcesFrom(payload);

    addBubble("assistant", answer, sources);

    if (!state.historyByDocument[state.selectedDocument]) {
      state.historyByDocument[state.selectedDocument] = [];
    }

    state.historyByDocument[state.selectedDocument].unshift({
      question,
      answer,
      sources,
      time: new Date().toLocaleString()
    });

    state.historyByDocument[state.selectedDocument] =
      state.historyByDocument[state.selectedDocument].slice(0, 50);

    saveHistory();
    renderHistory();
    setStatus("Ready", "ready");
  } catch (error) {
    addBubble("assistant", `I could not answer safely.\n\n${error.message}`);
    setStatus("Backend error", "error");
  } finally {
    $("askBtn").disabled = false;
  }
}

$("themeBtn").addEventListener("click", () => {
  setTheme(state.theme === "dark" ? "light" : "dark");
});

$("uploadBtn").addEventListener("click", uploadFile);

$("clearHistoryBtn").addEventListener("click", () => {
  if (!state.selectedDocument) return;
  delete state.historyByDocument[state.selectedDocument];
  saveHistory();
  renderHistory();
});

$("chatForm").addEventListener("submit", ask);

$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("chatForm").requestSubmit();
  }
});

setTheme(state.theme);
renderDocuments();
renderHistory();
checkBackend();

