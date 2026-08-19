"use strict";

const elements = {
  brandName: document.querySelector("#brand-name"),
  modelChip: document.querySelector("#model-chip"),
  contextChip: document.querySelector("#context-chip"),
  thinkButton: document.querySelector("#think-button"),
  saveButton: document.querySelector("#save-button"),
  sessionStatus: document.querySelector("#session-status"),
  budgetValue: document.querySelector("#budget-value"),
  budgetFill: document.querySelector("#budget-fill"),
  tickWarning: document.querySelector("#tick-warning"),
  tickProbe: document.querySelector("#tick-probe"),
  budgetNote: document.querySelector("#budget-note"),
  cumulativeUsage: document.querySelector("#cumulative-usage"),
  conversation: document.querySelector("#conversation"),
  empty: document.querySelector("#empty-state"),
  emptyName: document.querySelector("#empty-name"),
  scrollBottom: document.querySelector("#scroll-bottom"),
  pulse: document.querySelector("#pulse"),
  phaseText: document.querySelector("#phase-text"),
  phaseStats: document.querySelector("#phase-stats"),
  composer: document.querySelector("#composer"),
  input: document.querySelector("#message"),
  sendButton: document.querySelector("#send-button"),
  stopButton: document.querySelector("#stop-button"),
  sessionLabel: document.querySelector("#session-label"),
  charCount: document.querySelector("#char-count"),
  limitDialog: document.querySelector("#limit-dialog"),
  limitMessage: document.querySelector("#limit-message"),
  toastRoot: document.querySelector("#toast-root"),
  template: document.querySelector("#message-template"),
};

const numberFmt = new Intl.NumberFormat("en-US");

const TURN_STATUS = {
  complete: { label: "完成", tone: "ok" },
  truncated: { label: "截断", tone: "warn" },
  interrupted: { label: "已中断", tone: "warn" },
  failed: { label: "失败", tone: "error" },
  no_answer: { label: "无回答", tone: "warn" },
  blocked: { label: "已阻止", tone: "error" },
  pending: { label: "等待中", tone: "muted" },
};

const SESSION_STATUS = {
  active: { label: "active", tone: "ok" },
  paused: { label: "paused", tone: "warn" },
};

const state = {
  session: null,
  busy: true,
  follow: true,
  live: null,
  liveTokens: null,
  clockTimer: null,
  pollTimer: null,
  startedAt: 0,
  budget: { inputTokens: null, inputBudget: null, exact: false },
};

/* ---------- 格式化 ---------- */

function fmt(value) {
  return value == null ? "—" : numberFmt.format(value);
}

function fmtUsage(usage) {
  if (!usage) return "";
  return `in ${fmt(usage.input_tokens)} · out ${fmt(usage.output_tokens)}`;
}

function shortId(value) {
  if (!value) return "—";
  return value.length > 10 ? `${value.slice(0, 8)}…` : value;
}

/* ---------- 阶段状态与计时 ---------- */

function setPhase(text, active = false) {
  elements.phaseText.textContent = text;
  elements.pulse.classList.toggle("active", active);
}

function startClock() {
  state.startedAt = performance.now();
  updateClock();
  state.clockTimer = setInterval(updateClock, 120);
}

function updateClock() {
  const parts = [];
  if (state.liveTokens != null) parts.push(`${fmt(state.liveTokens)} tok`);
  const seconds = (performance.now() - state.startedAt) / 1000;
  parts.push(`${seconds.toFixed(1)}s`);
  elements.phaseStats.textContent = parts.join(" · ");
}

function stopClock(finalText) {
  if (state.clockTimer == null) return;
  clearInterval(state.clockTimer);
  state.clockTimer = null;
  if (finalText !== undefined) elements.phaseStats.textContent = finalText;
}

/* ---------- toast ---------- */

function toast(message, kind = "info") {
  const node = document.createElement("div");
  node.className = "toast";
  node.dataset.kind = kind;
  node.textContent = message;
  elements.toastRoot.append(node);
  setTimeout(() => {
    node.classList.add("leaving");
    setTimeout(() => node.remove(), 240);
  }, 3600);
}

/* ---------- busy / composer ---------- */

function setBusy(busy) {
  state.busy = busy;
  elements.input.disabled = busy;
  elements.sendButton.classList.toggle("hidden", busy);
  elements.stopButton.classList.toggle("hidden", !busy);
  elements.thinkButton.disabled = busy;
  elements.saveButton.disabled = busy;
  updateSendState();
}

function updateSendState() {
  elements.sendButton.disabled = state.busy || !elements.input.value.trim();
}

function resizeInput() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 200)}px`;
}

function updateCharCount() {
  const length = elements.input.value.length;
  elements.charCount.textContent = length > 0 ? `${numberFmt.format(length)} 字` : "";
}

/* ---------- 上下文预算 ---------- */

function renderBudgetTicks() {
  const budget = state.session?.budget;
  if (!budget || !budget.input_budget) return;
  const warnPct = Math.min(100, (budget.warning_threshold / budget.input_budget) * 100);
  const probePct = Math.min(100, (budget.probe_threshold / budget.input_budget) * 100);
  elements.tickWarning.style.left = `${warnPct}%`;
  elements.tickProbe.style.left = `${probePct}%`;
  elements.tickWarning.title = `warning 阈值 ${fmt(budget.warning_threshold)} tok`;
  elements.tickProbe.title = `probe 阈值 ${fmt(budget.probe_threshold)} tok`;
}

function setBudgetUsage(inputTokens, inputBudget, exact) {
  state.budget = { inputTokens, inputBudget, exact };
  updateBudgetFill();
}

function updateBudgetFill() {
  const budget = state.session?.budget;
  const { inputTokens, inputBudget, exact } = state.budget;
  if (inputTokens == null || !inputBudget) {
    elements.budgetFill.style.width = "0%";
    elements.budgetFill.dataset.stage = "ok";
    elements.budgetValue.textContent = "—";
    return;
  }
  const pct = Math.max(0.5, Math.min(100, (inputTokens / inputBudget) * 100));
  elements.budgetFill.style.width = `${pct}%`;
  elements.budgetValue.textContent = `${fmt(inputTokens)} / ${fmt(inputBudget)} tok${exact ? "" : " · 估"}`;
  let stage = "ok";
  if (budget && inputTokens >= budget.probe_threshold) stage = "danger";
  else if (budget && inputTokens >= budget.warning_threshold) stage = "warn";
  elements.budgetFill.dataset.stage = stage;
}

/* ---------- 溯源 ---------- */

function appendWarning(container, text) {
  const item = document.createElement("div");
  item.className = "prov-warning";
  item.textContent = `⚠ ${text}`;
  container.append(item);
}

function provSection(label) {
  const root = document.createElement("section");
  root.className = "prov-section";
  const head = document.createElement("div");
  head.className = "prov-title";
  head.textContent = label;
  const body = document.createElement("div");
  body.className = "prov-body";
  root.append(head, body);
  return { root, body };
}

function renderProvenance(container, options = {}) {
  container.replaceChildren();
  const warnings = [...(options.warnings ?? [])];
  if (options.error) warnings.unshift(`错误：${options.error}`);
  if (options.unverifiedRealtime) warnings.push("实时信息未经验证（unverified realtime）");
  for (const warning of warnings) appendWarning(container, warning);

  const knowledge = options.knowledgeSources ?? [];
  if (knowledge.length) {
    const section = provSection(`知识来源 · ${knowledge.length}`);
    for (const source of knowledge) {
      const item = document.createElement("div");
      item.className = "prov-item";
      const title = document.createElement("strong");
      title.textContent = source.title || source.document_key;
      const sub1 = document.createElement("span");
      sub1.className = "prov-sub mono";
      sub1.textContent = `${source.document_key} · ${source.source_location}`;
      const sub2 = document.createElement("span");
      sub2.className = "prov-sub mono";
      sub2.textContent = `rev ${shortId(source.revision_id)} · span ${source.start_char}..${source.end_char} · sha ${String(source.span_sha256).slice(0, 8)}`;
      item.title = `revision ${source.revision_id}\nsource ${source.source_id}\nfetched ${source.fetched_at}`;
      item.append(title, sub1, sub2);
      section.body.append(item);
    }
    container.append(section.root);
  }

  const web = options.webSources ?? [];
  if (web.length) {
    const section = provSection(`实时来源 · ${web.length}`);
    for (const source of web) {
      const item = document.createElement("div");
      item.className = "prov-item";
      const kind = document.createElement("span");
      kind.className = "prov-kind mono";
      kind.textContent = source.kind;
      const link = document.createElement("a");
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = source.title || source.url;
      const round = document.createElement("span");
      round.className = "prov-sub mono";
      round.textContent = `第 ${source.round} 轮检索`;
      item.append(kind, link, round);
      section.body.append(item);
    }
    container.append(section.root);
  }
  container.classList.toggle("hidden", !container.childElementCount);
}

/* ---------- 消息 ---------- */

function createMessage(role, content, options = {}) {
  elements.empty.classList.add("hidden");
  const node = elements.template.content.firstElementChild.cloneNode(true);
  node.classList.add(role);
  if (options.enter) node.classList.add("enter");
  const avatar = node.querySelector(".msg-avatar");
  const roleLabel = node.querySelector(".msg-role");
  const badge = node.querySelector(".msg-badge");
  const meta = node.querySelector(".msg-meta");
  const thinking = node.querySelector(".thinking");
  const contentBox = node.querySelector(".msg-content");
  const provenance = node.querySelector(".provenance");

  if (role === "user") {
    avatar.textContent = "你";
    roleLabel.textContent = "你";
    contentBox.textContent = content;
  } else if (role === "error") {
    avatar.textContent = "!";
    roleLabel.textContent = "错误";
    contentBox.textContent = content;
  } else {
    const aiName = options.aiName ?? state.session?.ai_name ?? "Hippocampus";
    avatar.textContent = aiName.slice(0, 1).toUpperCase();
    roleLabel.textContent = aiName;
    if (options.html) contentBox.innerHTML = options.html;
    else contentBox.textContent = content;
  }

  if (options.status) {
    const info = TURN_STATUS[options.status] ?? { label: options.status, tone: "muted" };
    badge.textContent = info.label;
    badge.dataset.tone = info.tone;
    badge.classList.remove("hidden");
  }
  if (options.thinking) {
    thinking.classList.remove("hidden");
    thinking.querySelector(".thinking-body").textContent = options.thinking;
    thinking.querySelector(".thinking-count").textContent = `${numberFmt.format(options.thinking.length)} 字`;
  }
  renderProvenance(provenance, options);
  if (options.meta) meta.textContent = options.meta;

  elements.conversation.append(node);
  return { node, content: contentBox, thinking, provenance, meta, badge };
}

/* ---------- 会话渲染 ---------- */

function renderSession(session) {
  state.session = session;
  document.title = `${session.ai_name} · Hippocampus`;
  elements.brandName.textContent = session.ai_name;
  elements.emptyName.textContent = session.ai_name;
  elements.modelChip.textContent = session.model;
  elements.modelChip.title = `Ollama ${session.ollama_version} · 模型上下文 ${fmt(session.model_context_length)} tok`;
  elements.contextChip.textContent = `ctx ${fmt(session.budget.context_window)}`;
  elements.contextChip.title = `input 预算 ${fmt(session.budget.input_budget)} tok · 输出上限 ${fmt(session.budget.max_output_tokens)} tok`;
  elements.thinkButton.setAttribute("aria-checked", String(session.think));
  elements.thinkButton.title = `thinking ${session.think ? "on" : "off"}（点击切换）`;
  const status = SESSION_STATUS[session.status] ?? { label: session.status, tone: "muted" };
  elements.sessionStatus.textContent = status.label;
  elements.sessionStatus.dataset.tone = status.tone;
  elements.sessionLabel.textContent = `session ${shortId(session.id)} · Ollama ${session.ollama_version}`;
  elements.sessionLabel.title = session.id;
  elements.cumulativeUsage.textContent = `累计 ${fmtUsage(session.cumulative_usage)}`;
  elements.cumulativeUsage.title = `probe 累计 ${fmtUsage(session.cumulative_probe_usage)}`;

  renderBudgetTicks();
  const lastUsage = [...session.turns].reverse().find((turn) => turn.usage?.input_tokens != null)?.usage;
  if (lastUsage) setBudgetUsage(lastUsage.input_tokens, session.budget.input_budget, true);
  else updateBudgetFill();

  elements.conversation.querySelectorAll(".msg").forEach((node) => node.remove());
  for (const turn of session.turns) {
    createMessage("user", turn.user);
    if (turn.assistant_markdown) {
      createMessage("assistant", turn.assistant_markdown, {
        aiName: session.ai_name,
        html: turn.assistant_html,
        thinking: turn.thinking,
        status: turn.status,
        error: turn.error,
        knowledgeSources: turn.knowledge_sources,
        webSources: turn.web_sources,
        warnings: turn.warnings,
        unverifiedRealtime: turn.unverified_realtime,
        meta: fmtUsage(turn.usage),
      });
    } else if (turn.error) {
      createMessage("error", turn.error, {
        status: turn.status,
        warnings: turn.warnings,
        meta: fmtUsage(turn.usage),
      });
    }
  }
  elements.empty.classList.toggle("hidden", session.turns.length > 0);
  const windowNote =
    session.budget.active_context_start_index > 0
      ? ` · 活跃窗口自第 ${session.budget.active_context_start_index + 1} 轮起`
      : "";
  elements.budgetNote.textContent = session.turns.length
    ? `共 ${session.turns.length} 轮历史${windowNote}`
    : "发送第一条消息后开始统计";
  setBusy(session.busy);
  if (state.follow) scrollToBottom();
}

async function loadSession() {
  const response = await fetch("/api/session", { headers: { accept: "application/json" } });
  if (!response.ok) throw new Error(`加载会话失败（HTTP ${response.status}）`);
  renderSession(await response.json());
}

/* ---------- 滚动 ---------- */

function scrollToBottom(smooth = false) {
  elements.conversation.scrollTo({
    top: elements.conversation.scrollHeight,
    behavior: smooth ? "smooth" : "auto",
  });
}

elements.conversation.addEventListener("scroll", () => {
  const box = elements.conversation;
  state.follow = box.scrollHeight - box.scrollTop - box.clientHeight < 96;
  elements.scrollBottom.classList.toggle("hidden", state.follow);
});

elements.scrollBottom.addEventListener("click", () => {
  state.follow = true;
  elements.scrollBottom.classList.add("hidden");
  scrollToBottom(true);
});

/* ---------- API ---------- */

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

/* ---------- SSE ---------- */

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
          const fieldValue = line.slice(colon + 1).replace(/^ /, "");
          if (field === "event") eventName = fieldValue;
          if (field === "data") data.push(fieldValue);
        }
      }
    }
  }
}

function handleStreamEvent(name, payload) {
  const live = state.live;
  switch (name) {
    case "status":
      setPhase(payload.message, true);
      break;
    case "prepared": {
      const exactLabel = payload.exact ? "精确" : "估计";
      setBudgetUsage(payload.input_tokens, payload.input_budget, payload.exact);
      setPhase(`生成中 · ${exactLabel} input ${fmt(payload.input_tokens)}/${fmt(payload.input_budget)} tok`, true);
      elements.budgetNote.textContent = `本轮纳入 ${payload.included} 轮 · 省略 ${payload.omitted} 轮`;
      break;
    }
    case "thinking":
      if (!live) break;
      live.thinkingText += payload.text;
      live.thinking.classList.remove("hidden");
      live.thinking.open = true;
      live.thinking.querySelector(".thinking-body").textContent = live.thinkingText;
      live.thinking.querySelector(".thinking-count").textContent = `${numberFmt.format(live.thinkingText.length)} 字`;
      if (payload.live_tokens != null) state.liveTokens = payload.live_tokens;
      if (state.follow) scrollToBottom();
      break;
    case "content":
      if (!live) break;
      if (!live.sawContent) {
        live.sawContent = true;
        live.thinking.open = false;
      }
      live.markdownText += payload.text;
      live.content.textContent = live.markdownText;
      if (payload.live_tokens != null) state.liveTokens = payload.live_tokens;
      if (state.follow) scrollToBottom();
      break;
    case "usage":
      if (payload.live_tokens != null) state.liveTokens = payload.live_tokens;
      break;
    case "completed":
      if (payload.usage) {
        const reason = payload.done_reason ? ` · ${payload.done_reason}` : "";
        setPhase(`生成收尾 · ${fmtUsage(payload.usage)}${reason}`, true);
      }
      break;
    case "done": {
      if (!live) break;
      live.content.classList.remove("streaming");
      live.content.innerHTML = payload.html;
      const info = TURN_STATUS[payload.status] ?? { label: payload.status, tone: "muted" };
      live.badge.textContent = info.label;
      live.badge.dataset.tone = info.tone;
      live.badge.classList.remove("hidden");
      live.meta.textContent = fmtUsage(payload.usage);
      renderProvenance(live.provenance, {
        error: payload.error,
        warnings: payload.warnings,
        unverifiedRealtime: payload.unverified_realtime,
        knowledgeSources: payload.knowledge_sources,
        webSources: payload.web_sources,
      });
      const sessionStatus = SESSION_STATUS[payload.session_status];
      if (sessionStatus) {
        elements.sessionStatus.textContent = sessionStatus.label;
        elements.sessionStatus.dataset.tone = sessionStatus.tone;
      }
      setPhase(payload.error ? "完成（含错误）" : "完成");
      stopClock(fmtUsage(payload.usage));
      if (state.follow) scrollToBottom();
      break;
    }
    case "limit":
      elements.limitMessage.textContent = payload.message;
      if (!elements.limitDialog.open) elements.limitDialog.showModal();
      setPhase("等待上下文决策…", true);
      break;
    case "error":
      throw new Error(payload.message);
  }
}

async function sendMessage(message) {
  setBusy(true);
  setPhase("正在连接模型…", true);
  state.liveTokens = null;
  startClock();
  createMessage("user", message, { enter: true });
  const live = createMessage("assistant", "", { aiName: state.session?.ai_name, enter: true });
  live.content.classList.add("streaming");
  state.live = { ...live, markdownText: "", thinkingText: "", sawContent: false };
  state.follow = true;
  elements.scrollBottom.classList.add("hidden");
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
    const message2 = error instanceof Error ? error.message : String(error);
    state.live?.node.remove();
    state.live = null;
    createMessage("error", message2, { enter: true });
    setPhase("请求失败");
    stopClock("");
    toast(message2, "error");
  } finally {
    stopClock();
    state.live = null;
    setBusy(false);
    elements.input.focus();
    await loadSession().catch(() => {});
  }
}

/* ---------- 事件绑定 ---------- */

elements.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = elements.input.value.trim();
  if (!message || state.busy) return;
  elements.input.value = "";
  resizeInput();
  updateCharCount();
  updateSendState();
  void sendMessage(message);
});

elements.input.addEventListener("input", () => {
  resizeInput();
  updateCharCount();
  updateSendState();
});

elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});

elements.stopButton.addEventListener("click", async () => {
  try {
    const result = await api("/api/cancel");
    setPhase(result.message, true);
  } catch (error) {
    toast(error.message, "error");
  }
});

elements.saveButton.addEventListener("click", async () => {
  if (state.busy) return;
  try {
    const result = await api("/api/save");
    toast(result.message, "success");
  } catch (error) {
    toast(error.message, "error");
  }
});

elements.thinkButton.addEventListener("click", async () => {
  if (!state.session || state.busy) return;
  const next = !state.session.think;
  elements.thinkButton.setAttribute("aria-checked", String(next));
  try {
    const result = await api("/api/think", { enabled: next });
    toast(result.message, "success");
    await loadSession();
  } catch (error) {
    elements.thinkButton.setAttribute("aria-checked", String(!next));
    toast(error.message, "error");
  }
});

elements.limitDialog.addEventListener("close", async () => {
  const action = elements.limitDialog.returnValue || "end";
  try {
    await api("/api/decision", { action });
    setPhase(action === "continue" ? "正在裁剪上下文并继续…" : "正在暂停会话…", true);
  } catch (error) {
    toast(error.message, "error");
  }
});

/* ---------- 启动 ---------- */

function showFatal(error) {
  setPhase("连接失败");
  const box = document.createElement("div");
  box.className = "fatal";
  const message = document.createElement("p");
  message.textContent = `无法加载会话：${error.message ?? error}`;
  const retry = document.createElement("button");
  retry.className = "ghost-button";
  retry.type = "button";
  retry.textContent = "重试";
  retry.addEventListener("click", () => location.reload());
  box.append(message, retry);
  elements.conversation.append(box);
}

function startPolling() {
  if (state.pollTimer) return;
  setPhase("另一客户端正在生成…", true);
  state.pollTimer = setInterval(async () => {
    try {
      await loadSession();
      if (!state.session?.busy) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        setPhase("就绪");
        toast("后台生成已完成", "success");
      }
    } catch {
      /* 继续轮询 */
    }
  }, 2500);
}

async function boot() {
  try {
    await loadSession();
    if (state.session?.busy) startPolling();
    else {
      setPhase("就绪");
      elements.input.focus();
    }
  } catch (error) {
    showFatal(error);
  }
}

void boot();
