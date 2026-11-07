#!/bin/zsh
# Send one support ticket to the trigger aimed at the WorkAF factory's intake,
# the way a help desk's webhook would. The body names the reasoner that runs
# the job ("run_on"), so WorkAF files it as an ordinary job that
# support.handle_ticket runs once it is started.
# Usage: fire-workaf.sh ["ticket text"] [ticket-id] [run_on]
set -euo pipefail
HERE=${0:A:h}; source $HERE/env.sh
TEXT=${1:-"Hi, the ledger CLI's total is wrong after I add a refund. I ran: ledger coffee 450 refund -500 and it printed \"2 entries, total 0.-50\". I expected -0.50. The repository is $DEMO_REPO."}
ID=${2:-T-$(date +%s)}
RUN_ON=${3:-support.handle_ticket}
K=$(<$LOCAL/cp/secrets/cpkey)
# The trigger this engine added (state.json), not whichever an older engine left.
TRIG=$(ssh $ENGINE_HOST "cat $KIT/engine/state.json" 2>/dev/null | jq -r '.trigger.id // empty')
[[ -n $TRIG ]] || die "no generic_bearer trigger aimed at codeaf-factory; is the engine up?"
BODY=$(jq -nc --arg t "$TEXT" --arg r "$RUN_ON" '{title:"ledger total wrong after a refund", body:$t, run_on:$r}')
say "POST $CP_URL/sources/$TRIG  (ticket $ID, run_on $RUN_ON)"
curl -fsS -X POST "http://127.0.0.1:$CP_PORT/sources/$TRIG" -H 'Content-Type: application/json' \
  -H "X-Ticket-Id: $ID" -K <(print -r -- "header = \"Authorization: Bearer $(<$LOCAL/cp/secrets/intake_bearer)\"") \
  -d "$BODY"
echo
