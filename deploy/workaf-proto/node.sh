#!/bin/zsh
# An example node on $NODE_HOST (dumb), run from spark.
# Usage: node.sh start|stop|restart|status|prove|clean [workaf-support|workaf-research]
#
# THE NODE BINDS NO PORT ANYONE CAN REACH. It runs link only
# (AGENTFIELD_NODE_LINK_ONLY=1): the SDK dials the control plane's node link,
# registers http://<node>.nodelink, and serves every dispatch, nested call,
# cancel and steer that arrives down that link. Its own listener is an
# ephemeral loopback port, which `prove` checks nothing outside can reach.
#
# Secrets are piped straight from where they live into mode-600 files on the
# node box, never through argv or this terminal:
#   openrouter   api_key from mac-studio's ~/.codeaf/config.json (read only)
#   github       `gh auth token` on spark
#   cpkey        the control plane's API key
# `stop` leaves them; `clean` removes them with everything else.
set -euo pipefail
HERE=${0:A:h}; source $HERE/env.sh
EX=${2:-workaf-support}
case $EX in
  workaf-support) PORT=$NODE_PORT;;
  workaf-research) PORT=$((NODE_PORT + 1));;
  *) die "unknown example $EX";;
esac
R="\$HOME/$KIT"          # the kit directory, expanded on the node box
RN="$R/$EX"               # this node's own state

remote() { ssh -o BatchMode=yes $NODE_HOST "$@"; }

running() { remote "kill -0 \$(cat $RN/node.pid 2>/dev/null) 2>/dev/null" ; }

ship() {
  local s=$LOCAL/stage/darwin-arm64
  [[ -x $s/$EX && -x $s/codeaf ]] || die "missing binaries; run build.sh nodes codeaf"
  remote "mkdir -p $R/bin $R/secrets $RN/home $RN/work $R/logs; chmod 700 $R/secrets"
  # Never copy over a binary that may be running: write beside it, then rename.
  for b in $EX codeaf; do
    if [[ $(remote "shasum -a 256 $R/bin/$b 2>/dev/null | cut -c1-64") != $(sha256sum $s/$b | cut -c1-64) ]]; then
      say "shipping $b to $NODE_HOST"
      scp -q $s/$b $NODE_HOST:$KIT/bin/.$b.new && remote "mv -f $R/bin/.$b.new $R/bin/$b"
    fi
  done
}

secrets() {
  remote "test -s $R/secrets/openrouter" || {
    ssh -o BatchMode=yes $ENGINE_HOST 'python3 -c "import json,os;print(json.load(open(os.path.expanduser(\"~/.codeaf/config.json\")))[\"api_key\"])"' \
      | remote "umask 077; cat > $R/secrets/openrouter"
  }
  remote "test -s $R/secrets/github" || gh auth token | remote "umask 077; cat > $R/secrets/github"
  remote "umask 077; cat > $R/secrets/cpkey" < $LOCAL/cp/secrets/cpkey
  remote "chmod 600 $R/secrets/*; test -s $R/secrets/openrouter && test -s $R/secrets/github" || die "a secret came through empty"
  say "secrets in place on $NODE_HOST (mode 600)"
}

# codeaf_home writes the harness's codeaf home ($AGENTFIELD_HOME/codeaf, which
# the SDK points CODEAF_HOME at): every model row is $MODEL, so no tier falls
# back to a built-in default. The key comes in through OPENROUTER_API_KEY_FILE.
codeaf_home() {
  remote "mkdir -p $RN/home/codeaf && chmod 700 $RN/home/codeaf && umask 077 && cat > $RN/home/codeaf/config.json" <<JSON
{"model.talk": "$MODEL", "task.model": "$MODEL", "vision_model": "$MODEL",
 "models.tiers.low": "$MODEL", "models.tiers.reflex": "$MODEL"}
JSON
  say "codeaf home on $NODE_HOST: every model row $MODEL"
}

start() {
  ship; secrets; codeaf_home
  if running; then say "$EX already up on $NODE_HOST pid $(remote "cat $RN/node.pid")"; return; fi
  # LINK ONLY: the SDK serves every control-plane request down the link it
  # dials, listens on an ephemeral loopback port nobody else can reach, and
  # registers http://$EX.nodelink. No tunnel, no inbound port.
  remote "cd $RN/work; env -i HOME=\$HOME USER=\$USER PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin \
    AGENT_NODE_ID=$EX AGENTFIELD_NODE_LINK_ONLY=1 \
    AGENTFIELD_URL=$CP_URL AGENTFIELD_API_KEY_FILE=$R/secrets/cpkey \
    OPENROUTER_API_KEY_FILE=$R/secrets/openrouter GITHUB_TOKEN_FILE=$R/secrets/github \
    AGENTFIELD_HOME=$RN/home CODEAF_BIN=$R/bin/codeaf SUPPORT_MODEL=$MODEL RESEARCH_MODEL=$MODEL \
    SUPPORT_WORKDIR=$RN/work RESEARCH_WORKDIR=$RN/work TMPDIR=$RN/work \
    GOPATH=$RN/gopath GOCACHE=$RN/gocache GOMODCACHE=$RN/gopath/pkg/mod HOMEBREW_NO_AUTO_UPDATE=1 \
    nohup $R/bin/$EX > $R/logs/$EX.log 2>&1 < /dev/null & echo \$! > $RN/node.pid"
  local key=$(<$LOCAL/cp/secrets/cpkey)
  for i in {1..60}; do
    curl -fsS -H "X-API-Key: $key" http://127.0.0.1:$CP_PORT/api/v1/nodes/$EX 2>/dev/null | grep -q '"id"' && break
    sleep 0.5
  done
  say "$EX up on $NODE_HOST pid $(remote "cat $RN/node.pid"), registered with the control plane"
  prove
}

# prove shows every TCP listener the node's processes hold on the node box, and
# that nothing on the node box answers on the node's port from spark.
prove() {
  say "listeners held by $EX and its children on $NODE_HOST:"
  remote "p=\$(cat $RN/node.pid); pids=\$p,\$(pgrep -d, -P \$p || true); lsof -nP -a -iTCP -sTCP:LISTEN -p \${pids%,} 2>/dev/null | sed 's/^/    /' || true"
  local bad=0 ip p
  local ports=(${(f)"$(remote "p=\$(cat $RN/node.pid); pids=\$p,\$(pgrep -d, -P \$p || true); lsof -nP -a -iTCP -sTCP:LISTEN -p \${pids%,} 2>/dev/null" | awk 'NR>1{n=split($9,a,":"); print a[n]}' | sort -u)"})
  for ip in $(remote "ipconfig getifaddr en0 2>/dev/null; ipconfig getifaddr en1 2>/dev/null; /Applications/Tailscale.app/Contents/MacOS/Tailscale ip -4 2>/dev/null" | sort -u); do
    for p in $ports; do
      [[ -n $p ]] || continue
      if nc -z -w 3 $ip $p 2>/dev/null; then say "  REACHABLE from spark: $ip:$p"; bad=1; else say "  not reachable from spark: $ip:$p"; fi
    done
  done
  curl -fsS -H "X-API-Key: $(<$LOCAL/cp/secrets/cpkey)" http://127.0.0.1:$CP_PORT/api/v1/nodes/$EX 2>/dev/null | jq -c '{id, base_url, health_status}' || true
  (( bad == 0 )) || die "the node is reachable from the control-plane host"
}

stop() {
  remote "for f in node tunnel; do p=\$(cat $RN/\$f.pid 2>/dev/null) && kill \$p 2>/dev/null; rm -f $RN/\$f.pid; done; \
    pkill -f '$KIT/bin/codeaf' 2>/dev/null; true"
  say "$EX stopped on $NODE_HOST"
}

status() {
  running && say "$EX up on $NODE_HOST" || say "$EX down on $NODE_HOST"
}

case ${1:-} in
  start) start;; stop) stop;; restart) stop; sleep 1; start;; status) status;; prove) prove;;
  clean) EX=workaf-research RN="$R/workaf-research" stop; EX=workaf-support RN="$R/workaf-support" stop
         remote "chmod -R u+w $R 2>/dev/null; rm -rf $R; sed -i '' -E \"/^${CP_SSH_ADDR//./\\\\.} /d\" \$HOME/.ssh/known_hosts; true"
         say "removed ~/$KIT on $NODE_HOST (binaries, homes, clones, logs, secrets) and its known_hosts row for $CP_SSH_ADDR";;
  *) die "usage: node.sh start|stop|restart|status|prove|clean [workaf-support|workaf-research]";;
esac
