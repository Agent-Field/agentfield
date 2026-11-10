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

    engine-driver.py <codeaf> <home> <dir> <cp_url> <cpkey_file> <model> [<github_token_file> <repo>]

With a GitHub token it also finishes the factory's setup on <repo> (checkout
skipped: the factory clones a repository the first job that needs it), turns
the runner on, and adds a generic_bearer trigger on AgentField aimed at the
factory's intake, so a ticket posted to it is filed as a WorkAF job. A ctl
line with "reply": "<path>" gets its answer written to that path.
"""
import json, os, subprocess, sys, threading, time

codeaf, home, d, cp_url, key_file, model = sys.argv[1:7]
gh_file = sys.argv[7] if len(sys.argv) > 7 else ""
repo = sys.argv[8] if len(sys.argv) > 8 else ""
os.makedirs(d, exist_ok=True)
log = open(os.path.join(d, "engine.jsonl"), "a", buffering=1)
err = open(os.path.join(d, "engine.stderr"), "a", buffering=1)
events = open(os.path.join(d, "events.jsonl"), "a", buffering=1)
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
        if "event" in msg:
            # Every event, stamped as it arrives: the live evidence.
            events.write(json.dumps({"at": time.time(), "msg": msg}) + "\n")
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

if gh_file and repo:
    tok = open(gh_file).read().strip()
    call("factory.setup_connect", {"caller": "workaf-proto", "source": "token", "token": tok})
    del tok
    call("factory.setup_select", {"caller": "workaf-proto", "repos": [repo]})
    for _ in range(60):
        v = call("factory.setup_read", {"caller": "workaf-proto"})
        if v.get("state") == "checkouts":
            break
        time.sleep(1)
    call("factory.setup_skip", {"caller": "workaf-proto", "repo": repo})
    call("factory.setup_done", {"caller": "workaf-proto"})
    state(runner=call("factory.local_run", {"caller": "workaf-proto"}))
    trig = call("factory.agentfield_trigger_add", {"factory": factory, "source": "generic_bearer",
                "config": {"idempotency_header": "X-Ticket-Id"}, "secret_env_var": "SUPPORT_INTAKE_BEARER"})
    state(trigger=trig)

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
                try:
                    res, failure = call(req["method"], req.get("params", {})), None
                except SystemExit as e:
                    res, failure = None, str(e)
                log.write(json.dumps({"ctl": req["method"], "result": res, "error": failure}) + "\n")
                if req.get("reply"):
                    tmp = req["reply"] + ".tmp"
                    json.dump({"result": res, "error": failure}, open(tmp, "w"))
                    os.replace(tmp, req["reply"])
                if req["method"] == "factory.agentfield_status":
                    state(status=res)
            except Exception as e:  # a bad ctl line must never take the engine down
                log.write(json.dumps({"ctl_error": str(e)}) + "\n")
