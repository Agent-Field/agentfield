# Shared settings for the three-machine prototype. Sourced by every script.
# Every value can be overridden from the environment.
#
#   control plane   spark       (linux/arm64)  the one reachable thing
#   example node    dumb        (darwin/arm64) dials out, binds no reachable port
#   WorkAF engine   mac-studio  (darwin/arm64) dials out, binds no reachable port
#
# Run the scripts ON spark (the control-plane host); they reach the other two
# boxes over ssh.

: ${CP_HOST:=spark}
: ${NODE_HOST:=dumb}
: ${ENGINE_HOST:=mac-studio}

# How the other boxes reach spark. mac-studio is on another LAN and reaches
# spark only over the tailnet; dumb reaches both.
: ${CP_ADDR:=100.64.233.75}
# The address dumb uses for its outbound ssh to spark. The tailnet address is
# gated by Tailscale SSH's interactive check, so this is spark's LAN address.
: ${CP_SSH_ADDR:=192.168.2.177}
: ${CP_SSH_USER:=santosh}
: ${NODE_SSH_KEY:='$HOME/.ssh/id_fleet'}   # expanded on the node box

: ${CP_PORT:=18741}
: ${CP_URL:=http://$CP_ADDR:$CP_PORT}
# The node's own HTTP port, bound to 127.0.0.1 on the node box only.
: ${NODE_PORT:=18742}

# The codeaf the harness and the engine run.
: ${CODEAF_REPO:=$HOME/src/codeaf}
: ${CODEAF_REF:=origin/proto/af-watch}
: ${MODEL:=deepseek/deepseek-v4.1-flash}
# The repository the factory's setup selects, and a native job runs on.
: ${DEMO_REPO:=santoshkumarradha/factory-demo}

# One directory per box holds everything this kit creates; `clean` removes it.
: ${KIT:=af-workaf-proto}
LOCAL=$HOME/$KIT
# Scripts set HERE=${0:A:h} before sourcing this file.
AF_ROOT=${HERE:h:h}

say() { print -r -- "[$(date +%H:%M:%S)] $*"; }
die() { say "ERROR: $*" >&2; exit 1; }
