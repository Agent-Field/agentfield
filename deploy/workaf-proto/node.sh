#!/bin/zsh
# An example node on $NODE_HOST (dumb), run from spark.
# Usage: node.sh start|stop|restart|status|prove|clean [workaf-support|workaf-research]
#
# THE NODE BINDS NO PORT ANYONE CAN REACH. The Go SDK on this branch still
# takes its dispatches over HTTP (its node link carries only steer and cancel),
# so the node serves on 127.0.0.1:$NODE_PORT and the node box itself dials out
# an ssh reverse tunnel that makes that loopback port appear on spark's
# loopback. The control plane dispatches to http://127.0.0.1:$NODE_PORT on
# spark. Every connection is opened by the node box. When the SDK answers
# `http` deliveries on its link, drop the tunnel and register the base_url
# http://<node>.nodelink instead (see RUN.md).
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

tunnel() {
  remote "kill -0 \$(cat $RN/tunnel.pid 2>/dev/null) 2>/dev/null" && return
  remote "nohup ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
    -o ControlMaster=no -o ControlPath=none -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i $NODE_SSH_KEY \
    -R 127.0.0.1:$PORT:127.0.0.1:$PORT $CP_SSH_USER@$CP_SSH_ADDR \
    > $R/logs/$EX-tunnel.log 2>&1 < /dev/null & echo \$! > $RN/tunnel.pid"
  for i in {1..20}; do nc -z 127.0.0.1 $PORT 2>/dev/null && break; sleep 0.5; done
  say "tunnel up: $NODE_HOST dialed $CP_SSH_ADDR; spark 127.0.0.1:$PORT → $NODE_HOST 127.0.0.1:$PORT"
}

start() {
  ship; secrets
  if running; then say "$EX already up on $NODE_HOST pid $(remote "cat $RN/node.pid")"; tunnel; return; fi
  tunnel
  remote "cd $RN/work; env -i HOME=\$HOME USER=\$USER PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin \
    AGENT_NODE_ID=$EX AGENT_LISTEN=127.0.0.1:$PORT AGENT_CALLBACK_URL=http://127.0.0.1:$PORT \
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
  local bad=0 ip
  for ip in $(remote "ipconfig getifaddr en0 2>/dev/null; ipconfig getifaddr en1 2>/dev/null; /Applications/Tailscale.app/Contents/MacOS/Tailscale ip -4 2>/dev/null" | sort -u); do
    if nc -z -w 3 $ip $PORT 2>/dev/null; then say "  REACHABLE from spark: $ip:$PORT"; bad=1; else say "  not reachable from spark: $ip:$PORT"; fi
  done
  (( bad == 0 )) || die "the node is reachable from the control-plane host"
}

stop() {
  remote "for f in node tunnel; do p=\$(cat $RN/\$f.pid 2>/dev/null) && kill \$p 2>/dev/null; rm -f $RN/\$f.pid; done; \
    pkill -f '$KIT/bin/codeaf' 2>/dev/null; true"
  say "$EX stopped on $NODE_HOST"
}

status() {
  running && say "$EX up on $NODE_HOST" || say "$EX down on $NODE_HOST"
  remote "kill -0 \$(cat $RN/tunnel.pid 2>/dev/null) 2>/dev/null" && say "tunnel up" || say "tunnel down"
}

case ${1:-} in
  start) start;; stop) stop;; restart) stop; sleep 1; start;; status) status;; prove) prove;;
  clean) EX=workaf-research RN="$R/workaf-research" stop; EX=workaf-support RN="$R/workaf-support" stop
         remote "chmod -R u+w $R 2>/dev/null; rm -rf $R; sed -i '' -E \"/^${CP_SSH_ADDR//./\\\\.} /d\" \$HOME/.ssh/known_hosts; true"
         say "removed ~/$KIT on $NODE_HOST (binaries, homes, clones, logs, secrets) and its known_hosts row for $CP_SSH_ADDR";;
  *) die "usage: node.sh start|stop|restart|status|prove|clean [workaf-support|workaf-research]";;
esac
