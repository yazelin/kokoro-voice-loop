# kokoro-voice-loop

基於 **Kokoro-82M**（超輕量 8200 萬參數 StyleTTS2 語音模型）的極速對話語音迴圈。

```
麥克風 ──► whisper.cpp（本機）──► llmshare / Groq（雲端）──► Kokoro-82M ONNX（純 CPU）──► paplay
  arecord        語音轉文字                 生成簡答                     極速發音（<100ms）
```

---

## 核心特色與架構優勢

### 1. 0 MB 顯存（純 CPU ONNX 極速推論）
採用 ONNX Runtime CPU 引擎，模型體積僅 **300 MB**，完全不佔用 GPU 顯存，顯卡跑滿 100% 也不影響語音生成。

實測（RTX 4060 Laptop 8GB 的 CPU 側、10.45 秒參考音、暖機後三次取中位數，2026-08-28）：

| 輸入 | 合成耗時 | 音訊長度 | RTF |
| --- | --- | --- | --- |
| 3 字 | 0.47s | 0.96s | 0.49 |
| 23 字 | 1.53s | 5.55s | 0.28 |
| 66 字 | 3.95s | 15.51s | 0.25 |

模型載入 1.2 秒，記憶體峰值約 1.1 GB。**這裡原本寫 RTF 約 0.1，是高估，已依實測更正。**
另外 RTF 會被語速灌水：Kokoro 唸得比 F5-TTS 慢（同樣 66 字唸出 15.5 秒對 10.4 秒），分母大所以 RTF 好看。要比「使用者等多久」請看合成耗時那一欄。

### 2. 世界級英文發音品質（ElevenLabs 水準）
Kokoro 原生在數萬小時的英語有聲書（LibriTTS/LJSpeech）上訓練，其英文發音的自然度、重音節奏與呼吸感達到開源頂峰：
* `af_heart`：美式溫暖 Podcast / 說書女聲（鎮店之寶）
* `af_bella`：美式清脆商務女聲
* `bf_emma`：正統英式優雅腔（British Accent）
* `am_adam`：美式自然青年男聲

### 3. 雞尾酒音色調配（Voice Blending）
Kokoro 的聲學生態支援向量線性疊加（`Style Vector Blending`）：
$$\text{Style}_{\text{Target}} = w_1 \cdot \text{Style}_1 + w_2 \cdot \text{Style}_2 + \dots$$
* **預設 `jinn` 音色配方**：
  `0.55 * zf_xiaobei + 0.30 * zf_xiaoni + 0.15 * af_bella`（語速 1.05），融合知性清脆與自媒體開場親切感。

### 4. 完整台灣化發音過濾器（`taiwanize.py` + `misaki[zh]`）
針對原生中文缺乏聲調與洋腔問題，本專案完整實裝三層台灣化處理：
1. **中文四聲 G2P**：透過 `misaki[zh]` 產生帶聲調（陰平→、陽平↗、上聲↓、去聲↘）的中文音素。
2. **台灣國語平舌化（去捲舌）**：
   * 捲舌音 `ㄕ (sh)` → 平舌音 `ㄙ (s)`（`ʂ` → `s`）
   * 捲舌音 `ㄓ (zh)` → 平舌音 `ㄗ (z/ts)`（`ʈʂ` → `ts`）
   * 捲舌音 `ㄔ (ch)` → 平舌音 `ㄘ (c/tsʰ)`（`ʈʂʰ` → `tsʰ`）
3. **雙唇圓唇介音化（消除 `/v/` 齒唇音）**：
   * 強制將 `w` 開頭轉為純雙唇音 `u`（`微/問/我` → `uei/uən/uo`，解決發成大陸北方 `vei/ven` 的齒唇音問題）。
4. **兩岸字音與詞彙校正**：
   * `垃圾` → `勒瑟`、`我和你` → `我汗你`、`企業` → `氣業`、`品質` → `品直`、`星期` → `星旗`、`微糖` → `為糖`。

---

## 快速開始

### 1. 安裝與依賴建立
```bash
cd ~/kokoro-voice-loop
bash setup.sh
```

### 2. 啟動對話迴圈
```bash
# 預設直接以 Jinn 調配音色啟動
.venv/bin/python voice_loop.py

# 指定單一純中文音色（例如小貝）
.venv/bin/python voice_loop.py --voice zf_xiaobei

# 切換 Groq 超快 LLM 後端
.venv/bin/python voice_loop.py --backend groq
```

---

## 互動指令清單

在提示字元 `請按 Enter 錄音，或輸入指令` 下：

| 指令 | 說明 | 備註 |
| :--- | :--- | :--- |
| **`:say <問題>`** | **打字向 LLM 提問**（不開麥克風） | 會呼叫 LLM 並記錄到對話記憶（`history`） |
| **`:tts <文字>`** | **純文字發音測試** | 跳過 LLM、不進記憶，純聽 TTS 發音與校音 |
| **`:voice <音色>`** | 切換音色（輸入 `:voice` 看可用清單） | 支援 `jinn`、`zf_xiaobei`、`af_heart` 等 54 種音色 |
| **`:blend <v1> <v2> <w>`** | 自行調配兩種音色混合比例 | 例：`:blend zf_xiaobei zf_xiaoni 0.7` |
| **`:speed <倍率>`** | 調整語速（0.5 ~ 2.0） | 例：`:speed 1.1` |
| **`:backend <後端>`** | 切換 LLM 後端 | `llmshare` / `groq` / `local` |
| **`:len <字數>`** | 設定回答字數上限 | 預設 50 字 |
| **`:history`** | 查看當前對話歷史 | 顯示前幾輪問答 |
| **`:clear`** | 清空對話記憶 | 開始新話題 |
| **`:help`** | 顯示指令說明 | |
| **`:q`** 或 `Ctrl+C` | 離開程式 | 乾淨退出 |

---

## 技術組件

* **聲學引擎**：`kokoro-onnx` (v1.0, 82M parameters, 24kHz)
* **G2P 字轉音素**：`misaki[zh]` ＋ 台灣國語音素後處理器
* **語言自動判斷**：輸入中文自動走 `misaki+taiwanize`；輸入英文自動切換 `en-us`/`en-gb` 頂級發音。
* **STT 引擎**：Whisper small（掛載 `-nf` 關閉溫度回退、`-sns` 抑制非語音、`-nth 0.6` 靜音過濾）

---

## 授權

MIT © 林亞澤 (yazelin)
