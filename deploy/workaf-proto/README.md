# Three-machine prototype kit

Runs the AgentField control plane on **spark**, an example node on **dumb** and
the codeaf (WorkAF) engine on **mac-studio**. Run every script on spark. Each
script is idempotent (`start` twice leaves one copy running) and has `stop`
and `clean`. Everything lives under `~/af-workaf-proto` on each box, and
`down.sh` removes all of it.

```
  mac-studio: codeaf app ──link (ws, outbound)──►┐
                                                  CONTROL PLANE  spark 0.0.0.0:18741
  dumb: workaf-support ──link (ws, outbound)─────►┘   ▲
        127.0.0.1:18742 ◄── ssh -R, dialed by dumb ───┘ (dispatch, see below)
```

| Box | OS | Reaches spark on |
| --- | --- | --- |
| spark | linux/arm64 | itself; LAN 192.168.2.177, tailnet 100.64.233.75 |
| dumb | darwin/arm64 | LAN and tailnet; ssh only on LAN (tailnet ssh asks for a browser check) |
| mac-studio | darwin/arm64 | tailnet only (another LAN) |

## Run it

```sh
cd deploy/workaf-proto
./build.sh                      # CP (native, cgo), nodes and codeaf (darwin/arm64), ~4 min cold
./cp.sh start
./node.sh start                 # workaf-support; `node.sh start workaf-research` for the other
./engine.sh start               # optional: codeaf app on mac-studio, linked to the CP
./fire.sh                       # one ticket through the generic_bearer trigger
./down.sh                       # stop and remove everything on all three boxes
```

`node.sh prove` and `engine.sh prove` print every TCP listener the process
holds and try each one from spark on every address the box has.

## Why the node still has a loopback port

The Go SDK on `proto/harness-watch` takes dispatches over HTTP; its node link
carries only `steer` and `cancel`, and answers any other delivery
"this node takes … over HTTP". So the node binds `127.0.0.1` only, and dumb
dials an `ssh -R` to spark that puts the port on spark's loopback, where the
control plane reaches it. Every connection is opened by dumb. When the SDK
answers `http` deliveries on the link (the control plane's carrier for
`<node>.nodelink` hosts is already there), set
`AGENT_CALLBACK_URL=http://workaf-support.nodelink` and drop `tunnel()` in
`node.sh`.

## Secrets

Piped from where they live into mode-600 files, never through argv or a
terminal: the model key from mac-studio's `~/.codeaf/config.json` (read only),
the GitHub token from `gh auth token` on spark, and the control plane's own key
and the trigger's bearer, which `cp.sh` makes. `clean` deletes them.

The engine gets a fresh codeaf home whose `config.json` holds only `api_key`,
`setup_seen_at` and every model row set to `$MODEL`.

## Settings

`env.sh` has every host, address, port and the codeaf ref (`CODEAF_REF`,
default `origin/proto/af-watch`, which has `codeaf exec --events/--steer`).
Each can be overridden from the environment.
