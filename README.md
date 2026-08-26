# kokoro-voice-loop

基於 **Kokoro-82M**（超輕量 8200 萬參數語音模型）的極速對話語音迴圈。

```
麥克風 ──► whisper.cpp（本機）──► llmshare / Groq（雲端）──► Kokoro-82M ONNX（純 CPU）──► paplay
  arecord        語音轉文字                 生成簡答                     極速發音（<100ms）
```

---

## 核心特色

1. **零顯存佔用（純 CPU 執行）**：使用 ONNX Runtime CPU 引擎，推論速度極快（延遲 < 100ms），不與 GPU / LLM 搶顯存。
2. **`:say` 測試模式**：不需錄音，打字輸入 `:say 任何句子` 立即發音測試。
3. **`:blend` 語音調配**：支援將兩種不同性格/性別的音色進行加權混合（例如 `:blend zf_xiaobei zf_xiaoni 0.7`）。
4. **`:speed` 語速調節**：精準控制語速（0.5 ~ 2.0）。

---

## 快速開始

### 1. 安裝與下載模型
```bash
cd ~/kokoro-voice-loop
bash setup.sh
```

### 2. 啟動對話迴圈
```bash
.venv/bin/python voice_loop.py
```

### 3. 指令說明
在交談互動提示下輸入：
* `:say <文字>` — 直接測試發音（不錄音、不呼叫 LLM）
* `:voice <名稱>` — 切換音色（輸入 `:voice` 可看可用中文清單，如 `zf_xiaobei`, `zf_xiaoni`, `zm_yunjian`）
* `:blend <音色1> <音色2> <權重>` — 混合音色（例：`:blend zf_xiaobei zf_xiaoni 0.7`）
* `:speed <倍率>` — 調整語速（例：`:speed 1.1`）
* `:backend <llmshare/groq/local>` — 切換 LLM 後端
* `:clear` — 清空對話記憶
* `:help` — 顯示說明
* `:q` — 離開

---

## 授權

MIT © 林亞澤
