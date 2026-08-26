#!/usr/bin/env python3
"""麥克風 → Whisper → LLM → Kokoro-82M（純 CPU / 超輕量 ONNX）講出來 → 播放。

特點：
- 零顯存佔用（純 CPU 執行），推論極速（<100ms）
- 支援 :say 直接打字測試發音
- 支援 :blend 混合兩種音色與語氣（例：:blend zf_xiaobei zf_xiaoni 0.7）
- 支援 :speed 調節語速（0.5 ~ 2.0）

跑法：
  .venv/bin/python voice_loop.py
  .venv/bin/python voice_loop.py --voice zf_xiaobei
  .venv/bin/python voice_loop.py --selfcheck
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from taiwanize import taiwanize_text, taiwanize_phonemes

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models/kokoro-v1.0.onnx"
VOICES_PATH = HERE / "models/voices-v1.0.bin"
WORK = HERE / "tmp"

WHISPER_CLI = Path(os.environ.get("WHISPER_CLI", Path.home() / ".mori/bin/whisper-cli"))
WHISPER_MODEL = Path(os.environ.get("WHISPER_MODEL", Path.home() / ".mori/models/ggml-small.bin"))
WHISPER_DESCRIPTOR = Path.home() / ".mori/whisper-server.json"
WHISPER_SUPERVISOR = Path.home() / ".mori/bin/mori-whisper-serve"

LLM_URL = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "local": "http://127.0.0.1:8080/v1/chat/completions",
}
DEFAULT_MODEL = {
    "llmshare": "deepseek-v4-flash:0731",
    "groq": "openai/gpt-oss-120b",
    "local": "qwen3.5-4b",
}

PAREN_RE = re.compile(r"[(（\[][^)）\]]{0,6}[)）\]]")
STT_HINT = "以下是繁體中文的句子。"

COMMANDS = {
    ":voice": "切換音色，例：:voice zf_xiaobei（直接輸入 :voice 看可用清單）",
    ":blend": "混合兩種音色，例：:blend zf_xiaobei zf_xiaoni 0.7",
    ":speed": "調整語速 (0.5~2.0)，例：:speed 1.1",
    ":say": "不錄音，直接發音或測試，例：:say 今天天氣真好",
    ":backend": "換 LLM 後端：:backend llmshare / groq / local",
    ":len": "回答字數上限，例：:len 40",
    ":clear": "清空對話歷史",
    ":history": "查看對話紀錄",
    ":help": "顯示指令清單",
    ":q": "離開",
}


HALLUCINATIONS = {
    "謝謝大家收看", "謝謝大家收看。", "請訂閱我的頻道", "請訂閱我的頻道。",
    "請不吝賜教", "謝謝大家", "謝謝大家。", "未完待續", "感謝您的收看",
}


def clean_stt(text):
    text = PAREN_RE.sub("", text).strip()
    if not text or text in STT_HINT or text in HALLUCINATIONS:
        return ""
    if re.match(r"^(.{2,12}?)\1{2,}[。！!？\?]*$", text):
        return ""
    return text


def record(out_wav, device):
    cmd = ["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", str(out_wav)]
    if device:
        cmd[1:1] = ["-D", device]
    proc = subprocess.Popen(cmd)
    input("錄音中… 再按 Enter 停止。")
    proc.terminate()
    proc.wait()
    return out_wav.exists() and out_wav.stat().st_size > 16000


def find_whisper_server():
    try:
        d = json.loads(WHISPER_DESCRIPTOR.read_text(encoding="utf-8"))
        os.kill(d["pid"], 0)
    except (OSError, ValueError, KeyError):
        return None
    return f"http://{d['host']}:{d['port']}{d.get('inference_path', '/inference')}"


def ensure_whisper_server(timeout=20):
    url = find_whisper_server()
    if url or not WHISPER_SUPERVISOR.is_file():
        return url
    try:
        subprocess.run([str(WHISPER_SUPERVISOR), "--ensure"], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    deadline = time.time() + timeout
    while time.time() < deadline:
        url = find_whisper_server()
        if url:
            return url
        time.sleep(0.5)
    return None


def transcribe(wav, stt=None):
    if stt and stt.get("url"):
        r = subprocess.run(
            ["curl", "-s", "--max-time", "10", "-F", f"file=@{wav}", "-F", "language=zh",
             "-F", "response_format=json", "-F", f"prompt={STT_HINT}", stt["url"]],
            capture_output=True, text=True,
        )
        try:
            return clean_stt(" ".join(json.loads(r.stdout)["text"].split()))
        except (ValueError, KeyError):
            pass

    env = {**os.environ, "LD_LIBRARY_PATH": str(WHISPER_CLI.parent)}
    r = subprocess.run(
        [str(WHISPER_CLI), "-m", str(WHISPER_MODEL), "-l", "zh", "-nt", "-np",
         "-nf", "-sns", "-nth", "0.6", "--prompt", STT_HINT, "-f", str(wav)],
        capture_output=True, text=True, env=env,
    )
    if r.returncode != 0:
        return ""
    return clean_stt(r.stdout)


def build_prompt(question, max_chars, history=()):
    rule = f"用正體中文口語回答，{max_chars} 個字以內，只回答問題本身，不要開場白、不要條列、不要 emoji。"
    if not history:
        return rule + f"問題：{question}"
    past = "\n".join(f"我：{q}\n你：{a}" for q, a in history[-8:])
    return f"{rule}下面是我們剛才的對話，接著回答最後那個問題：\n\n{past}\n我：{question}\n你："


def ask_llm(question, backend, model, max_chars, history=()):
    prompt = build_prompt(question, max_chars, history)
    if backend == "llmshare":
        r = subprocess.run(["llmshare", "raw", model, prompt], capture_output=True, text=True)
        raw = r.stdout if r.returncode == 0 else f"模型無回應: {r.stderr.strip()[:60]}"
    elif backend == "groq":
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            return "未設定 GROQ_API_KEY"
        payload = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 300})
        req = urllib.request.Request(LLM_URL["groq"], payload.encode(),
                                     {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "kokoro-loop/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
            raw = data["choices"][0]["message"].get("content") or ""
        except Exception as e:
            raw = f"Groq 錯誤: {e}"
    else:
        payload = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 300})
        req = urllib.request.Request(LLM_URL["local"], payload.encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
            raw = data["choices"][0]["message"].get("content") or ""
        except Exception as e:
            raw = f"在地模型錯誤: {e}"
    return " ".join(raw.split())[:max_chars * 2] or "我不太確定該怎麼回答。"


PRESETS = {
    "jinn": {
        "desc": "Jinn（調配風格：55% zf_xiaobei + 30% zf_xiaoni + 15% af_bella）",
        "weights": [("zf_xiaobei", 0.55), ("zf_xiaoni", 0.30), ("af_bella", 0.15)],
        "speed": 1.05,
    }
}


def selfcheck():
    assert MODEL_PATH.is_file(), f"找不到模型檔 {MODEL_PATH}"
    assert VOICES_PATH.is_file(), f"找不到音色檔 {VOICES_PATH}"
    import kokoro_onnx
    k = kokoro_onnx.Kokoro(str(MODEL_PATH), str(VOICES_PATH))
    voices = k.get_voices()
    assert len(voices) > 0, "無可用音色"
    samples, sr = k.create("測試", voice="zf_xiaobei", lang="cmn")
    assert len(samples) > 0 and sr == 24000
    print("Kokoro selfcheck OK! (Available voices: %d)" % len(voices))


def taiwanize_phonemes(p):
    # 1. 台灣國語平舌化（去捲舌音）
    p = p.replace("ʂ", "s")       # ㄕ -> ㄙ (sh -> s)
    p = p.replace("ʈʂʰ", "tsʰ")   # ㄔ -> ㄘ (ch -> c)
    p = p.replace("ʈʂ", "ts")     # ㄓ -> ㄗ (zh -> z)
    p = p.replace("ʐ", "z")       # ㄖ -> 軟化 (r -> z)
    # 2. 雙唇圓唇介音化（去除北方齒唇音 v / vei，轉為純雙唇 ㄨ [u] / uei, uən）
    p = p.replace("w", "u")       # 微/問/我 -> ㄨㄟ, ㄨㄣ, ㄨㄛ
    return p


def main():
    ap = argparse.ArgumentParser(description="Kokoro-82M Voice Loop")
    ap.add_argument("--voice", default="jinn", help="預設音色，支援 jinn 或 zf_xiaobei, zf_xiaoni, zm_yunjian")
    ap.add_argument("--speed", type=float, default=None, help="語速 (預設 jinn=1.05, 一般=1.0)")
    ap.add_argument("--backend", choices=["llmshare", "groq", "local"], default="llmshare")
    ap.add_argument("--model", help="指定 LLM 模型")
    ap.add_argument("--max-chars", type=int, default=50, help="回答最大字數")
    ap.add_argument("--device", default="", help="arecord 裝置")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()

    if not MODEL_PATH.is_file() or not VOICES_PATH.is_file():
        sys.exit(f"模型檔尚未就緒，請先確認 {MODEL_PATH} 與 {VOICES_PATH} 存在。")

    WORK.mkdir(parents=True, exist_ok=True)
    import kokoro_onnx
    import soundfile as sf

    print("載入 Kokoro-82M ONNX 引擎...", flush=True)
    t0 = time.time()
    kokoro = kokoro_onnx.Kokoro(str(MODEL_PATH), str(VOICES_PATH))
    all_voices = kokoro.get_voices()
    zh_voices = [v for v in all_voices if v.startswith("z")]
    print(f"Kokoro 載入完成（{time.time()-t0:.2f}s）！中文音色：{', '.join(zh_voices)}")

    def resolve_voice(vname):
        if vname in PRESETS:
            p = PRESETS[vname]
            style = sum(w * kokoro.get_voice_style(v) for v, w in p["weights"])
            spd = p["speed"]
            return vname, style, spd
        if vname in all_voices:
            return vname, None, 1.0
        return "zf_xiaobei", None, 1.0

    v_name, v_style, v_spd = resolve_voice(args.voice)
    speed = args.speed if args.speed is not None else v_spd

    model = args.model or DEFAULT_MODEL[args.backend]
    state = {
        "voice": v_name,
        "custom_style": v_style,
        "speed": speed,
        "backend": args.backend,
        "model": model,
        "len": args.max_chars,
    }
    history = []
    stt = {"url": ensure_whisper_server()}

    print(f"後端：{state['backend']} / {state['model']} | 音色：{state['voice']} | 語速：{state['speed']}")
    print("提示：按 Enter 錄音，打字輸入 :say <文字> 直接測試，打 :help 看完整指令。\n")

    from misaki.zh import ZHG2P
    g2p = ZHG2P()

    out_wav = WORK / "out.wav"
    in_wav = WORK / "in.wav"

    while True:
        try:
            line = input("請按 Enter 錄音，或輸入指令（:say / :help）: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再見！")
            break

        if line == ":q":
            break
        if line == ":help":
            print("\n指令清單：")
            for k, v in COMMANDS.items():
                print(f"  {k:9s} {v}")
            print()
            continue
        if line == ":clear":
            history.clear()
            print("對話歷史已清空。\n")
            continue
        if line == ":history":
            if not history:
                print("尚無對話歷史。\n")
            for q, a in history[-8:]:
                print(f"  問：{q}\n  答：{a}")
            print()
            continue
        if line.startswith(":speed"):
            _, _, val = line.partition(" ")
            try:
                state["speed"] = max(0.5, min(2.5, float(val)))
                print(f"語速調整為：{state['speed']}\n")
            except ValueError:
                print("請給數字，例：:speed 1.1\n")
            continue
        if line.startswith(":voice"):
            _, _, vname = line.partition(" ")
            vname = vname.strip()
            if not vname:
                presets_str = ", ".join(f"{k} ({v['desc']})" for k, v in PRESETS.items())
                print(f"\n推薦預設調配音：{presets_str}\n原生中文音色: {', '.join(zh_voices)}\n全部音色: {', '.join(all_voices)}\n")
                continue
            if vname in PRESETS:
                v_name, v_style, v_spd = resolve_voice(vname)
                state["voice"] = v_name
                state["custom_style"] = v_style
                state["speed"] = v_spd
                print(f"音色已切換為：{v_name}（{PRESETS[vname]['desc']}）\n")
            elif vname in all_voices:
                state["voice"] = vname
                state["custom_style"] = None
                print(f"音色已切換為：{vname}\n")
            else:
                print(f"找不到音色 {vname}，請打 :voice 查看清單。\n")
            continue
        if line.startswith(":blend"):
            parts = line.split()
            if len(parts) >= 4:
                v1, v2, w = parts[1], parts[2], parts[3]
                try:
                    weight = float(w)
                    s1 = kokoro.get_voice_style(v1)
                    s2 = kokoro.get_voice_style(v2)
                    state["custom_style"] = (weight * s1) + ((1.0 - weight) * s2)
                    state["voice"] = f"blend({v1}*{weight} + {v2}*{1-weight:.2f})"
                    print(f"已調配混合音色：{state['voice']}\n")
                except Exception as e:
                    print(f"混合失敗：{e}\n")
            else:
                print("格式錯誤，例：:blend zf_xiaobei zf_xiaoni 0.7\n")
            continue
        if line.startswith(":backend"):
            _, _, b = line.partition(" ")
            b = b.strip()
            if b in DEFAULT_MODEL:
                state["backend"] = b
                state["model"] = DEFAULT_MODEL[b]
                print(f"後端切換為 {b} / {state['model']}\n")
            else:
                print("可選後端：llmshare, groq, local\n")
            continue

        typed_say = ""
        if line.startswith(":say"):
            _, _, typed_say = line.partition(" ")
            typed_say = typed_say.strip()
            if not typed_say:
                print("請給要測試的句子，例：:say 你好！\n")
                continue

        turn_start = time.time()
        turn_start = time.time()
        if typed_say:
            heard = typed_say
            print(f"測試文字：{heard}")
            answer_display = taiwanize_text(heard, for_speech=False)
            speech_text = taiwanize_text(heard, for_speech=True)
            stt_time = 0.0
            llm_time = 0.0
        else:
            if not record(in_wav, args.device):
                print("未錄到聲音，請再試一次。\n")
                continue
            t_stt = time.time()
            heard = transcribe(in_wav, stt)
            stt_time = time.time() - t_stt
            print(f"你說：{heard}（STT {stt_time:.2f}s）")
            if not heard:
                print("聽不出內容，請再試一次。\n")
                continue
            t_llm = time.time()
            raw_answer = ask_llm(heard, state["backend"], state["model"], state["len"], history)
            llm_time = time.time() - t_llm
            answer_display = taiwanize_text(raw_answer, for_speech=False)
            speech_text = taiwanize_text(raw_answer, for_speech=True)
            print(f"回答：{answer_display}（LLM {llm_time:.2f}s）")

        t_tts = time.time()
        v_target = state["custom_style"] if state["custom_style"] is not None else state["voice"]
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", speech_text))
        try:
            if has_cjk:
                # 使用 misaki[zh] 產生帶四聲調的中文音素，並經過台灣國語去捲舌與雙唇化轉換
                raw_phonemes, _ = g2p(speech_text)
                phonemes = taiwanize_phonemes(raw_phonemes)
                samples, sr = kokoro.create(phonemes, voice=v_target, speed=state["speed"], is_phonemes=True)
                lang_tag = "zh (taiwanize+misaki)"
            else:
                lang_code = "en-gb" if isinstance(state["voice"], str) and state["voice"].startswith("b") else "en-us"
                if state["custom_style"] is None and isinstance(v_target, str) and v_target.startswith("z"):
                    v_target = "af_heart"
                samples, sr = kokoro.create(speech_text, voice=v_target, speed=state["speed"], lang=lang_code)
                lang_tag = lang_code

            sf.write(str(out_wav), samples, sr)
            tts_time = time.time() - t_tts
            audio_sec = len(samples) / sr
            print(f"Kokoro 合成（{lang_tag}）：{tts_time:.2f}s | 音訊長：{audio_sec:.1f}s | 總耗時：{time.time()-turn_start:.2f}s")
            subprocess.run(["paplay", str(out_wav)])
            if not typed_say:
                history.append((heard, answer_display))
            print()
        except Exception as e:
            print(f"Kokoro 發音失敗: {e}\n")


if __name__ == "__main__":
    main()
