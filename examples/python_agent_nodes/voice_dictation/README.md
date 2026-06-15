# Voice Dictation — AgentField realtime session demo

A minimal, runnable demo of the realtime **session** DX from PR #654: a
Wispr-Flow-style live dictation page where speech is transcribed in real time
through an AgentField session.

```
 ┌─────────┐  SDP offer (application/sdp)   ┌──────────────┐   SDP + server key   ┌──────────────┐
 │ Browser │ ────────────────────────────▶ │ Control plane │ ───────────────────▶ │ OpenAI       │
 │  (mic)  │ ◀──────────────────────────── │  (SDP proxy)  │ ◀─────────────────── │  Realtime    │
 └────┬────┘        answer SDP              └──────────────┘     answer SDP        └──────┬───────┘
      │                                                                                   │
      └───────────────────────────  audio media (WebRTC, direct)  ─────────────────────────┘
```

The control plane only sits in **signaling** — it proxies the SDP exchange using
its own server-side `OPENAI_API_KEY`. Audio media flows **directly**
browser ↔ OpenAI. The browser never holds a provider credential. This is exactly
OpenAI's documented "unified interface" pattern.

## Prerequisites

- The AgentField control plane built from this branch (`af server`).
- `OPENAI_API_KEY` exported **for the control plane process**, with **Realtime
  API access** (this is a paid/gated capability on your OpenAI account).
- A browser with mic access. On WSL, use the **Windows-side browser** (see below).

## Run it

**1. Start the control plane** with your key (separate shell):

```bash
export OPENAI_API_KEY=sk-...
af server                       # listens on :8080
```

**2. (Optional) Register this agent** so the session declaration + `save_note`
tool exist in the control plane and `af session start` works:

```bash
cd examples/python_agent_nodes/voice_dictation
pip install -r requirements.txt
af run                          # or: python main.py
```

> The live transcription demo does **not** require the agent — `realtime-offer`
> is self-contained. The agent is here to show the `@app.session(...)`
> declaration DX and the `tools=[...]` allowlist routing through `execute/async`.

**3. Serve the page on an allowed CORS origin** (the control plane allows
`http://localhost:5173` by default):

```bash
cd examples/python_agent_nodes/voice_dictation/web
python -m http.server 5173
```

**4. Open it.** On WSL, open **`http://localhost:5173` in your Windows browser**
(Chrome/Edge). WSL2 forwards `localhost`, so the Windows browser reaches both the
page (`:5173`) and the control plane (`:8080`) running in WSL, while WebRTC audio
goes straight from the Windows browser to OpenAI. Click **Start dictation**,
allow the mic, and talk — your words stream into the box. **Copy** puts the text
on your clipboard.

## WSL notes / constraints

- **Mic + browser must be on the Windows side.** WSL has no audio device; running
  a Linux browser inside WSL won't capture your mic. The Windows browser is the
  client — that's fine, because in this architecture the client is *supposed* to
  be the untrusted edge.
- `http://localhost` is a secure context, so `getUserMedia` works without HTTPS.
- If the cross-origin POST is blocked, serve the page on `:5173` (default allowed
  origin) or set `AGENTFIELD_API_CORS_ALLOWED_ORIGINS` to include your origin.

## The "paste job" (Wispr-Flow-style global dictation) — optional, Windows-side

True global dictation (hotkey → transcribe → paste into the focused app) needs
OS-level keyboard + clipboard integration, which **can't** be done from inside
WSL — it has to run on Windows. The **Copy** button is the pragmatic substitute.
If you want the full hotkey flow, run a tiny [AutoHotkey](https://www.autohotkey.com/)
script **on Windows** that pastes the clipboard on a chord:

```ahk
; paste-dictation.ahk  (Windows, AutoHotkey v2)
; Win+Shift+V : type out whatever the dictation page copied to the clipboard
#+v:: {
    SendText A_Clipboard
}
```

A fuller version would talk to the page over a WebSocket and auto-paste on each
committed utterance, but that's beyond this demo's scope.

## How this maps to PR #654

- ✅ **SDP proxy** — `POST /api/v1/sessions/:id/realtime-offer` (the validated path).
- ✅ **Key boundary** — provider key stays on the control plane.
- ✅ **`tools=[...]` allowlist** — `save_note` routes via `execute/async` into the DAG.
- ⚠️ **Input transcription / VAD** — the page sends a client-side `session.update`
  because session-level config isn't exposed yet (issue **#663**).
- ⚠️ **Live handler loop** — `session.input()/say()` in `main.py` are scaffolding
  until issue **#664** lands; the browser drives the validated flow today.
