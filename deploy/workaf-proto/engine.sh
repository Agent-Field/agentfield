#!/bin/zsh
# The codeaf (WorkAF) engine on $ENGINE_HOST (mac-studio), run from spark.
# Usage: engine.sh start|stop|restart|status|prove|clean
#
# It runs `codeaf app` against a FRESH codeaf home under ~/af-workaf-proto on
# mac-studio and NEVER touches ~/.codeaf, which it only reads for api_key. Every
# model row in the fresh home is $MODEL. engine-driver.py stands in for the
# desk: it starts the local factory and connects it to the control plane with
# no base_url, so the factory dials out over the node link and binds no port
# for AgentField.
set -euo pipefail
HERE=${0:A:h}; source $HERE/env.sh
R="\$HOME/$KIT"
E="$R/engine"

remote() { ssh -o BatchMode=yes $ENGINE_HOST "$@"; }
running() { remote "kill -0 \$(cat $E/driver.pid 2>/dev/null) 2>/dev/null"; }

ship() {
  local s=$LOCAL/stage/darwin-arm64
  [[ -x $s/codeaf ]] || die "missing codeaf; run build.sh codeaf"
  remote "mkdir -p $R/bin $R/secrets $E/home $R/logs; chmod 700 $R/secrets $E/home"
  if [[ $(remote "shasum -a 256 $R/bin/codeaf 2>/dev/null | cut -c1-64") != $(sha256sum $s/codeaf | cut -c1-64) ]]; then
    say "shipping codeaf to $ENGINE_HOST"
    scp -q $s/codeaf $ENGINE_HOST:$KIT/bin/.codeaf.new && remote "mv -f $R/bin/.codeaf.new $R/bin/codeaf"
  fi
  scp -q $HERE/engine-driver.py $ENGINE_HOST:$KIT/bin/engine-driver.py
}

# home writes the fresh home's config.json: api_key copied from ~/.codeaf,
# every model row set to $MODEL. ~/.codeaf is opened read-only.
home() {
  remote "python3 - '$MODEL'" <<'PY'
import json, os, sys
model = sys.argv[1]
src = json.load(open(os.path.expanduser("~/.codeaf/config.json")))
dst_dir = os.path.expanduser("~/af-workaf-proto/engine/home")
cfg = {"api_key": src["api_key"], "setup_seen_at": src.get("setup_seen_at", ""),
       "model.talk": model, "task.model": model, "vision_model": model,
       "models.tiers.low": model, "models.tiers.reflex": model}
path = os.path.join(dst_dir, "config.json")
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as f:
    json.dump({k: v for k, v in cfg.items() if v != ""}, f, indent=2)
print("fresh home config rows:", ", ".join(sorted(k for k in cfg if k != "api_key")))
PY
  remote "umask 077; cat > $R/secrets/cpkey" < $LOCAL/cp/secrets/cpkey
}

start() {
  ship
  if running; then say "engine already up on $ENGINE_HOST"; status; return; fi
  home
  remote "rm -f $E/state.json $E/ctl; cd $E; env -i HOME=\$HOME USER=\$USER PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin \
    CODEAF_HOME=$E/home CODEAF_NO_UPDATE_CHECK=1 CODEAF_TELEMETRY=off TMPDIR=$E \
    nohup /usr/bin/python3 $R/bin/engine-driver.py $R/bin/codeaf $E/home $E $CP_URL $R/secrets/cpkey $MODEL \
    > $R/logs/engine-driver.log 2>&1 < /dev/null & echo \$! > $E/driver.pid"
  for i in {1..90}; do
    remote "grep -q '\"link\": \"up\"' $E/state.json 2>/dev/null" && break
    running || { remote "tail -20 $R/logs/engine-driver.log $E/engine.stderr"; die "the engine driver exited"; }
    sleep 1
  done
  status
  prove
}

status() {
  running && say "engine up on $ENGINE_HOST (driver pid $(remote "cat $E/driver.pid"), codeaf app pid $(remote "cat $E/engine.pid 2>/dev/null"))" || say "engine down on $ENGINE_HOST"
  remote "cat $E/state.json 2>/dev/null" | jq -c '{factory, link: .status.link, link_ups: .status.link_ups, listening: .status.listening, node: .status.node_id}' 2>/dev/null || true
  local key=$(<$LOCAL/cp/secrets/cpkey)
  curl -fsS -H "X-API-Key: $key" http://127.0.0.1:$CP_PORT/api/v1/nodes/codeaf-factory 2>/dev/null | jq -c '{id, base_url, health_status}' || true
}

# prove lists every TCP listener codeaf app holds on mac-studio and checks none
# answers from spark.
prove() {
  say "listeners held by codeaf app on $ENGINE_HOST:"
  local out
  out=$(remote "p=\$(cat $E/engine.pid); lsof -nP -a -iTCP -sTCP:LISTEN -p \$p,\$(pgrep -d, -P \$p || echo \$p) 2>/dev/null" || true)
  print -r -- "${out:-    (none)}" | sed 's/^/    /'
  local ports=(${(f)"$(print -r -- $out | awk 'NR>1{n=split($9,a,":"); print a[n]}' | sort -u)"})
  local bad=0 ip p
  for ip in $(remote "ipconfig getifaddr en0 2>/dev/null; ipconfig getifaddr en1 2>/dev/null; /Applications/Tailscale.app/Contents/MacOS/Tailscale ip -4 2>/dev/null" | sort -u); do
    for p in $ports; do
      [[ -n $p ]] || continue
      if nc -z -w 3 $ip $p 2>/dev/null; then say "  REACHABLE from spark: $ip:$p"; bad=1; else say "  not reachable from spark: $ip:$p"; fi
    done
  done
  (( bad == 0 )) || die "the engine is reachable from the control-plane host"
}

stop() {
  remote "for f in driver engine; do p=\$(cat $E/\$f.pid 2>/dev/null) && kill \$p 2>/dev/null; done; sleep 1; \
    pkill -f '$KIT/bin/codeaf' 2>/dev/null; rm -f $E/driver.pid $E/engine.pid $E/ctl; true"
  say "engine stopped on $ENGINE_HOST"
}

case ${1:-} in
  start) start;; stop) stop;; restart) stop; sleep 1; start;; status) status;; prove) prove;;
  clean) stop; remote "rm -rf $R"; say "removed ~/$KIT on $ENGINE_HOST (binary, fresh home, logs, secrets); ~/.codeaf untouched";;
  *) die "usage: engine.sh start|stop|restart|status|prove|clean";;
esac
