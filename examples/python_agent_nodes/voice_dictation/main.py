"""Voice dictation agent — AgentField realtime session demo (PR #654 flow).

This is a minimal, runnable example of the realtime *session* DX:

    @app.session("voice", provider="openai", transport="webrtc", ...)

The browser captures the mic, negotiates a WebRTC peer connection, and the
AgentField control plane proxies the SDP offer/answer to OpenAI Realtime using
its *server-side* API key (the browser never holds a provider credential). Audio
flows directly browser <-> OpenAI; the control plane only sits in signaling.

The `tools=[...]` list is the allowlist the live model is permitted to invoke
during the session. Here we expose `save_note`, which routes back through the
normal AgentField execute/async path (so the work shows up in the workflow DAG)
via `POST /api/v1/sessions/:id/tools/save_note`.

Run:
    export OPENAI_API_KEY=sk-...           # must have Realtime access
    af server                              # control plane on :8080 (separate shell)
    af run                                 # or: python main.py  -> registers this agent
    # then serve web/ and open the page (see README.md)
"""

import os

from agentfield import Agent, AIConfig
from pydantic import BaseModel

app = Agent(
    node_id="voice-dictation",
    agentfield_server=os.getenv("AGENTFIELD_URL", "http://localhost:8080"),
    ai_config=AIConfig(
        model=os.getenv("SMALL_MODEL", "openai/gpt-4o-mini"), temperature=0.2
    ),
)


class SavedNote(BaseModel):
    ok: bool
    chars: int
    preview: str


@app.reasoner()
async def save_note(text: str) -> SavedNote:
    """Persist a dictated note (session-scoped run-memory).

    This is the kind of thing the live voice model can call autonomously because
    it's listed in `tools=[...]` below. When the control plane routes a session
    tool call here, it arrives as a normal AgentField execution carrying the
    session id, so it lands in the workflow DAG like any other reasoner.
    """
    preview = text.strip().replace("\n", " ")[:80]
    # Store in run-scoped memory so it's visible on the session's workflow.
    await app.memory.set("run", "last_note", text)
    return SavedNote(ok=True, chars=len(text), preview=preview)


@app.session(
    "voice",
    provider="openai",
    transport="webrtc",
    model=os.getenv("REALTIME_MODEL", "gpt-realtime"),
    modalities=["audio", "text"],
    voice=os.getenv("REALTIME_VOICE", "marin"),
    tools=["voice-dictation.save_note"],
)
async def voice(session):
    """Handler-side orchestration for the voice session.

    NOTE (PR #654): `session.input()` / `session.say()` are still scaffolding in
    the SDK — the live handler loop is tracked in issue #664. The *validated*
    path that this demo exercises is the SDP proxy + tool routing, which the
    browser drives directly. This handler is here to show the declaration DX and
    will become the live loop once #664 lands.
    """
    turn = await session.input()
    result = await session.call("voice-dictation.save_note", text=turn.transcript or turn.text or "")
    await session.say(f"Saved {result['chars']} characters.")


if __name__ == "__main__":
    app.run()
