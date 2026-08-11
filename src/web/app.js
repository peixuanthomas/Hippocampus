"use strict";

const elements = {
  conversation: document.querySelector("#conversation"),
  empty: document.querySelector("#empty-state"),
  composer: document.querySelector("#composer"),
  input: document.querySelector("#message"),
  send: document.querySelector("#send-button"),
  cancel: document.querySelector("#cancel-button"),
  save: document.querySelector("#save-button"),
  think: document.querySelector("#think-button"),
  model: document.querySelector("#model-chip"),
  context: document.querySelector("#context-chip"),
  sessionStatus: document.querySelector("#session-status"),
  runtime: document.querySelector("#runtime-message"),
  token: document.querySelector("#token-status"),
  pulse: document.querySelector("#pulse"),
  sessionLabel: document.querySelector("#session-label"),
  template: document.querySelector("#message-template"),
  limitDialog: document.querySelector("#limit-dialog"),
  limitMessage: document.querySelector("#limit-message"),
};

let state = { session: null, busy: false, live: null };

function setRuntime(message, active = false) {
  elements.runtime.textContent = message;
  elements.pulse.classList.toggle("active", active);
}

function setBusy(busy) {
  state.busy = busy;
  elements.input.disabled = busy;
  elements.send.disabled = busy;
  elements.cancel.classList.toggle("hidden", !busy);
}

function formatUsage(usage) {
  if (!usage) return "";
  const input = usage.input_tokens ?? "未知";
  const output = usage.output_tokens ?? "未知";
  return `input ${input} · output ${output}`;
}

function createMessage(role, content, options = {}) {
  elements.empty?.classList.add("hidden");
  const node = elements.template.content.firstElementChild.cloneNode(true);
  node.classList.add(role);
  const icon = node.querySelector(".role-icon");
  const label = node.querySelector(".role-label");
  const markdown = node.querySelector(".markdown");
  const thinking = node.querySelector(".thinking");
  const meta = node.querySelector(".message-meta");
  if (role === "user") {
    icon.textContent = "›";
    label.textContent = "You";
    markdown.textContent = content;
  } else if (role === "error") {
    icon.textContent = "!";
    label.textContent = "Error";
    markdown.textContent = content;
  } else {
    icon.textContent = "◆";
    label.textContent = "Hippocampus";
    if (options.html) markdown.innerHTML = options.html;
    else markdown.textContent = content;
  }
  if (options.thinking) {
    thinking.classList.remove("hidden");
    thinking.querySelector("pre").textContent = options.thinking;
  }
  meta.textContent = options.meta ?? "";
  elements.conversation.append(node);
  return { node, markdown, thinking, meta };
}

function renderSession(session) {
  state.session = session;
  elements.model.textContent = session.model;
  elements.think.textContent = `think:${session.think ? "on" : "off"}`;
  elements.context.textContent = `ctx:${session.budget.context_window}`;
  elements.sessionStatus.textContent = session.status;
  elements.sessionLabel.textContent = `session ${session.id} · Ollama ${session.ollama_version}`;
  elements.conversation.querySelectorAll(".message-row").forEach((node) => node.remove());
  for (const turn of session.turns) {
    createMessage("user", turn.user);
    if (turn.assistant_markdown) {
      createMessage("assistant", turn.assistant_markdown, {
        html: turn.assistant_html,
        thinking: turn.thinking,
        meta: `${turn.status} · ${formatUsage(turn.usage)}`,
      });
    } else if (turn.error) {
      createMessage("error", `[${turn.status}] ${turn.error}`);
    }
  }
  elements.empty.classList.toggle("hidden", session.turns.length > 0);
  elements.token.textContent = `累计 ${formatUsage(session.cumulative_usage)}`;
  setBusy(session.busy);
  setRuntime(session.busy ? "已有任务正在运行…" : "就绪", session.busy);
  scrollToBottom(false);
}

async function loadSession() {
  const response = await fetch("/api/session", { headers: { accept: "application/json" } });
  if (!response.ok) throw new Error(`加载会话失败：${response.status}`);
  renderSession(await response.json());
}

function scrollToBottom(smooth = true) {
  window.scrollTo({ top: document.documentElement.scrollHeight, behavior: smooth ? "smooth" : "auto" });
}

function resizeInput() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 220)}px`;
}

async function api(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  const payload = await response.json().catch(() => ({ message: `HTTP ${response.status}` }));
  if (!response.ok) throw new Error(payload.message ?? `HTTP ${response.status}`);
  return payload;
}

async function sendMessage(message) {
  setBusy(true);
  setRuntime("正在连接模型…", true);
  elements.token.textContent = "";
  createMessage("user", message);
  const live = createMessage("assistant", "…");
  live.markdown.textContent = "";
  state.live = { ...live, markdownText: "", thinkingText: "" };
  scrollToBottom();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json", accept: "text/event-stream" },
      body: JSON.stringify({ message }),
    });
    if (!response.ok || !response.body) {
      const payload = await response.json().catch(() => ({ message: `HTTP ${response.status}` }));
      throw new Error(payload.message ?? `HTTP ${response.status}`);
    }
    await readEventStream(response.body, handleStreamEvent);
  } catch (error) {
    state.live?.node.remove();
    createMessage("error", error instanceof Error ? error.message : String(error));
    setRuntime("请求失败");
  } finally {
    state.live = null;
    setBusy(false);
    elements.input.focus();
    await loadSession().catch(() => {});
  }
}

async function readEventStream(body, handler) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventName = "message";
  let data = [];
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replaceAll("\r\n", "\n");
    while (buffer.includes("\n")) {
      const index = buffer.indexOf("\n");
      const line = buffer.slice(0, index);
      buffer = buffer.slice(index + 1);
      if (line === "") {
        if (data.length) handler(eventName, JSON.parse(data.join("\n")));
        eventName = "message";
        data = [];
      } else if (!line.startsWith(":")) {
        const colon = line.indexOf(":");
        if (colon >= 0) {
          const field = line.slice(0, colon);
          const value = line.slice(colon + 1).replace(/^ /, "");
          if (field === "event") eventName = value;
          if (field === "data") data.push(value);
        }
      }
    }
  }
}

function handleStreamEvent(name, payload) {
  const live = state.live;
  if (name === "status") setRuntime(payload.message, true);
  if (name === "prepared") {
    const source = payload.exact ? "精确" : "估计";
    setRuntime(`生成中 · ${source} input ${payload.input_tokens ?? "未知"}/${payload.input_budget}`, true);
    elements.token.textContent = `included ${payload.included} · omitted ${payload.omitted}`;
  }
  if (name === "thinking" && live) {
    live.thinkingText += payload.text;
    live.thinking.classList.remove("hidden");
    live.thinking.querySelector("pre").textContent = live.thinkingText;
  }
  if (name === "content" && live) {
    live.markdownText += payload.text;
    live.markdown.textContent = live.markdownText;
    live.meta.textContent = `live output ${payload.live_tokens ?? "—"}`;
    scrollToBottom();
  }
  if (name === "completed") {
    elements.token.textContent = formatUsage(payload.usage);
  }
  if (name === "done" && live) {
    live.markdown.innerHTML = payload.html;
    live.meta.textContent = `${payload.status} · ${formatUsage(payload.usage)}`;
    elements.sessionStatus.textContent = payload.session_status;
    setRuntime(payload.error ?? "完成");
  }
  if (name === "limit") {
    elements.limitMessage.textContent = payload.message;
    if (!elements.limitDialog.open) elements.limitDialog.showModal();
    setRuntime("等待上下文决策…", true);
  }
  if (name === "error") {
    throw new Error(payload.message);
  }
}

elements.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = elements.input.value.trim();
  if (!message || state.busy) return;
  elements.input.value = "";
  resizeInput();
  void sendMessage(message);
});

elements.input.addEventListener("input", resizeInput);
elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});

elements.cancel.addEventListener("click", async () => {
  try {
    const result = await api("/api/cancel");
    setRuntime(result.message, true);
  } catch (error) {
    setRuntime(error.message);
  }
});

elements.save.addEventListener("click", async () => {
  try {
    const result = await api("/api/save");
    setRuntime(result.message);
  } catch (error) {
    setRuntime(error.message);
  }
});

elements.think.addEventListener("click", async () => {
  if (!state.session || state.busy) return;
  try {
    const result = await api("/api/think", { enabled: !state.session.think });
    setRuntime(result.message);
    await loadSession();
  } catch (error) {
    setRuntime(error.message);
  }
});

elements.limitDialog.addEventListener("close", async () => {
  const action = elements.limitDialog.returnValue || "end";
  try {
    await api("/api/decision", { action });
    setRuntime(action === "continue" ? "正在裁剪上下文…" : "正在暂停会话…", true);
  } catch (error) {
    setRuntime(error.message);
  }
});

loadSession().catch((error) => setRuntime(error.message));
