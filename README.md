# Hippocampus

Hippocampus 是一个面向本地 Ollama 的单会话终端聊天工具。它只保存一个会话的原始 JSON 记录与可审计的上下文预算轨迹；没有长期记忆、检索、向量库、摘要或跨会话拼接。

要求 Python >= 3.11。安装后先自行启动并准备模型（工具不会自动下载）：

```bash
python -m venv .venv
.venv/bin/pip install -e '.[test]'
ollama serve
ollama pull qwen3.5:9b
```

使用：

```bash
hippocampus new
hippocampus --sessions-dir ./sessions new --model qwen3.5:9b --no-think
hippocampus new --context-window 32768 --max-output-tokens 4096 --safety-margin-tokens 512
hippocampus new --system-prompt '请简洁回答'
hippocampus new --system-prompt-file prompt.txt
hippocampus list
hippocampus show 20260811-abcdef12
hippocampus resume 20260811-abcdef12
python -m hippocampus --help
```

默认上下文窗口为 32768，输出预留 4096，安全余量 512，因此输入预算为 28160。达到输入预算的 80% 时会进行精确 probe；达到 90% 时会要求选择“继续”或“结束”。继续时只丢弃最旧的完整轮次，并展示保留/舍弃数量。`/budget` 可查看当前与最近 trace。

聊天内命令：`/budget`、`/think on|off`、`/save`、`/help`、`/exit`。thinking 和正文分通道实时显示；thinking 会保存在 JSON 中，但不会被再次注入模型上下文。每个会话独立保存为 `sessions/<id>.json`，可用唯一前缀恢复。

Token 语义：流中的 live 输出仅是暂时计数，最终事件的 `input/output/total` 才是权威值。probe 的开销与回答开销分开累计。中断前没有最终事件时，未知项明确显示“未知”，不会虚构计数。

默认测试不会访问 Ollama：

```bash
pytest -q
```

真实集成测试需要服务和模型：

```bash
HIPPOCAMPUS_RUN_OLLAMA_INTEGRATION=1 pytest -q tests/test_integration_ollama.py
```
