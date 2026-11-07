#!/usr/bin/env python3
"""Stand in for the WorkAF desk: run `codeaf app`, connect its factory to the
control plane over the node link, and keep it running.

The desk normally owns `codeaf app`'s stdin. With no desk on the box, this
driver owns it instead: it starts the local factory, sets its model, connects
it to AgentField with NO base_url (link only, so the factory binds no port for
AgentField), then forwards any line written to the `ctl` fifo as a call, so
other scripts can poke the running engine:

    echo '{"method":"factory.agentfield_status","params":{"factory":"<id>"}}' > ctl

Every answer and event goes to engine.jsonl; state.json holds the factory id
and the last agentfield_status. Usage:

    engine-driver.py <codeaf> <home> <dir> <cp_url> <cpkey_file> <model>
"""
import json, os, subprocess, sys, threading, time

codeaf, home, d, cp_url, key_file, model = sys.argv[1:7]
os.makedirs(d, exist_ok=True)
log = open(os.path.join(d, "engine.jsonl"), "a", buffering=1)
err = open(os.path.join(d, "engine.stderr"), "a", buffering=1)
proc = subprocess.Popen([codeaf, "app", "--home", home], stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1)
with open(os.path.join(d, "engine.pid"), "w") as f:
    f.write(str(proc.pid))

pending, lock, wlock, nxt = {}, threading.Lock(), threading.Lock(), [0]


def reader():
    for line in proc.stdout:
        log.write(line)
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if "id" in msg and "event" not in msg:
            with lock:
                box = pending.pop(msg["id"], None)
            if box is not None:
                box.append(msg)


threading.Thread(target=reader, daemon=True).start()


def call(method, params, wait=180):
    with lock:
        nxt[0] += 1
        i = nxt[0]
        box = []
        pending[i] = box
    with wlock:
        proc.stdin.write(json.dumps({"id": i, "method": method, "params": params}) + "\n")
        proc.stdin.flush()
    end = time.time() + wait
    while not box and time.time() < end and proc.poll() is None:
        time.sleep(0.1)
    if not box:
        raise SystemExit(f"{method}: no answer")
    if box[0].get("error"):
        raise SystemExit(f"{method}: {box[0]['error']}")
    return box[0].get("result")


def state(**kv):
    path = os.path.join(d, "state.json")
    cur = {}
    if os.path.exists(path):
        cur = json.load(open(path))
    cur.update(kv)
    json.dump(cur, open(path + ".tmp", "w"), indent=2)
    os.replace(path + ".tmp", path)


local = call("factory.local_start", {"caller": "workaf-proto"})
factory = local["factory"]
state(factory=factory, local=local)
call("factory.spending_set", {"factory": factory, "default_model": model})
key = open(key_file).read().strip()
st = call("factory.agentfield_connect", {"factory": factory, "url": cp_url, "api_key": key})
del key
state(connect=st)
for _ in range(30):
    st = call("factory.agentfield_status", {"factory": factory})
    if st.get("link") == "up":
        break
    time.sleep(1)
state(status=st)

ctl = os.path.join(d, "ctl")
if not os.path.exists(ctl):
    os.mkfifo(ctl, 0o600)
while proc.poll() is None:
    with open(ctl) as f:  # blocks until a writer opens it
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                res = call(req["method"], req.get("params", {}))
                log.write(json.dumps({"ctl": req["method"], "result": res}) + "\n")
                if req["method"] == "factory.agentfield_status":
                    state(status=res)
            except (ValueError, KeyError, SystemExit) as e:
                log.write(json.dumps({"ctl_error": str(e)}) + "\n")
