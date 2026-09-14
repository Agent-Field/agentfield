# Session turn detection and interruption

OpenAI sessions accept `turn_detection` in Python and TypeScript, and
`WithSessionTurnDetection` in Go. It is supported with explicit
`provider="openai"` and `transport="webrtc"` or `"websocket"`. Supplying it for
OpenRouter `audio_turns` is an error; AgentField does not switch providers or
transports.

Omitting the configuration enables interruptible server VAD:

```json
{
  "type": "server_vad",
  "threshold": 0.5,
  "prefix_padding_ms": 300,
  "silence_duration_ms": 500,
  "create_response": true,
  "interrupt_response": true
}
```

These defaults also apply to older OpenAI registrations without this field.
An explicitly supplied object must specify `type`. Missing optional fields use
mode-specific defaults; explicit `false` and `0` are preserved.

## Python

```python
from agentfield import Agent

app = Agent("support")

@app.session(
    "voice",
    provider="openai",
    transport="webrtc",
    turn_detection={
        "type": "server_vad",
        "threshold": 0.6,
        "silence_duration_ms": 700,
        "interrupt_response": True,
    },
)
async def voice(session):
    pass
```

`ServerVAD`, `SemanticVAD`, and the `TurnDetection` union are exported for type
annotations. The decorator validates the dictionary when the session is declared.

## TypeScript

```typescript
import { Agent } from '@agentfield/sdk';

const app = new Agent({ nodeId: 'support' });
app.session('voice', {
  provider: 'openai',
  transport: 'webrtc',
  turn_detection: {
    type: 'semantic_vad',
    eagerness: 'low',
    interrupt_response: true
  }
}, async (session) => {});
```

`TurnDetection` is an exported discriminated union. Runtime validation also
rejects invalid values supplied by JavaScript or external configuration.

## Go

```go
interrupt := false
silence := 700
err := app.RegisterSession("voice", "openai", "webrtc",
    agent.WithSessionTurnDetection(agent.TurnDetection{
        Type:              "server_vad",
        SilenceDurationMS: &silence,
        InterruptResponse: &interrupt,
    }),
)
```

Optional numeric and boolean fields are pointers so an unset field can be
distinguished from an explicit zero or false. `RegisterSession` returns a
validation error before updating the registry.

## Supported options

| Option | Modes | Default | Validation |
| --- | --- | --- | --- |
| `type` | Both | `server_vad` when config is omitted | `server_vad` or `semantic_vad`; required in an explicit object |
| `threshold` | Server | `0.5` | Finite number from 0 to 1 |
| `prefix_padding_ms` | Server | `300` | Non-negative integer milliseconds |
| `silence_duration_ms` | Server | `500` | Non-negative integer milliseconds |
| `eagerness` | Semantic | `auto` | `auto`, `low`, `medium`, or `high` |
| `create_response` | Both | `true` | Boolean; automatically respond after a detected turn |
| `interrupt_response` | Both | `true` | Boolean; interrupt an ongoing response when speech starts |

Server-only fields cannot be supplied with semantic VAD, and `eagerness` cannot
be supplied with server VAD. Unknown fields, invalid values, and null field values
are rejected. Setting both response flags to `false` keeps speech detection
active while leaving response creation and cancellation to the client.

## Control-plane connection

Start the registered session with
`POST /api/v1/session-targets/<node>.<session>/start`, then POST the raw SDP offer
to the returned `offer_url` with `Content-Type: application/sdp`. Preserve its
query parameters: they identify the registered target, provider, transport,
model, and voice. The start response also includes the resolved `turn_detection`.

The offer endpoint re-reads the registered target and validates its configuration
before contacting OpenAI. It sends the resolved options under
`session.audio.input.turn_detection` in the multipart session configuration.
This is a stateless lookup, so a registration change between start and offer is
reflected when the offer is submitted. There is no session database migration.

The CLI can select the same registered configuration:

```sh
agentfield session offer <session_id> --provider openai --transport webrtc \
  --target support.voice --sdp @offer.sdp
```

Legacy direct offers without a target keep working and use the interruptible
server-VAD defaults. Clients constructing offer URLs themselves must include
`target=<node>.<session>` to use author-defined settings. The existing offer
endpoint remains WebRTC-only; accepting WebSocket registration metadata does
not add a WebSocket connection adapter.

See [OpenAI's VAD guide](https://developers.openai.com/api/docs/guides/realtime-vad)
for the provider's turn detection and interruption behavior.
