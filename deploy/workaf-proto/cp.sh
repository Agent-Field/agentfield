#!/bin/zsh
# The control plane on spark: SQLite under ~/af-workaf-proto/cp/home, listening
# on 0.0.0.0:$CP_PORT, because it is the one thing the other boxes dial.
# Usage: cp.sh start|stop|restart|status|clean
#   start   idempotent: a running control plane is left alone
#   stop    kills it (its database is kept)
#   clean   stop, then remove its database, secrets and logs
set -euo pipefail
HERE=${0:A:h}; source $HERE/env.sh
D=$LOCAL/cp
BIN=$LOCAL/stage/linux-arm64/agentfield-server

running() { [[ -f $D/pid ]] && kill -0 $(<$D/pid) 2>/dev/null; }

secrets() {
  mkdir -p $D/secrets && chmod 700 $D/secrets
  # cpkey: the control plane's API key. intake_bearer: the shared token the
  # support ticket webhook (generic_bearer) checks. Made once, kept until clean.
  for f in cpkey intake_bearer; do
    [[ -s $D/secrets/$f ]] || (umask 077; openssl rand -hex 24 > $D/secrets/$f)
  done
}

start() {
  [[ -x $BIN ]] || die "no $BIN; run build.sh cp first"
  if running; then say "control plane already up pid $(<$D/pid) at $CP_URL"; return; fi
  secrets; mkdir -p $D/home $LOCAL/logs
  AGENTFIELD_HOME=$D/home AGENTFIELD_STORAGE_MODE=local AGENTFIELD_PORT=$CP_PORT \
  AGENTFIELD_API_KEY=$(<$D/secrets/cpkey) SUPPORT_INTAKE_BEARER=$(<$D/secrets/intake_bearer) \
    nohup $BIN server >> $LOCAL/logs/cp.log 2>&1 < /dev/null &
  print $! > $D/pid
  for i in {1..60}; do
    curl -fsS http://127.0.0.1:$CP_PORT/api/v1/health >/dev/null 2>&1 && { say "control plane up pid $(<$D/pid) at $CP_URL"; return; }
    sleep 0.5
  done
  die "control plane did not come up; see $LOCAL/logs/cp.log"
}

stop() {
  if running; then kill $(<$D/pid); for i in {1..20}; do running || break; sleep 0.25; done; running && kill -9 $(<$D/pid); fi
  rm -f $D/pid; say "control plane stopped"
}

status() {
  if running; then say "control plane up pid $(<$D/pid)"; else say "control plane down"; fi
  ss -ltnp 2>/dev/null | grep ":$CP_PORT " || true
}

case ${1:-} in
  start) start;; stop) stop;; restart) stop; sleep 1; start;; status) status;;
  clean) stop; rm -rf $D $LOCAL/logs/cp.log; say "removed $D";;
  *) die "usage: cp.sh start|stop|restart|status|clean";;
esac
