#!/usr/bin/env python3
"""One call into the running engine through engine-driver.py's ctl fifo:

    rpc.py <engine dir> <method> '<params json>'   → prints {"result", "error"}
"""
import json, os, sys, time, uuid

d, method = os.path.abspath(sys.argv[1]), sys.argv[2]
params = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
reply = os.path.join(d, "rpc-" + uuid.uuid4().hex)
with open(os.path.join(d, "ctl"), "w") as f:
    f.write(json.dumps({"method": method, "params": params, "reply": reply}) + "\n")
for _ in range(2000):
    if os.path.exists(reply):
        out = open(reply).read()
        os.remove(reply)
        print(out)
        sys.exit(0)
    time.sleep(0.1)
print(json.dumps({"error": "no answer in 200s"}))
sys.exit(1)
