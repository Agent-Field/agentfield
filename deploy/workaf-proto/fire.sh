#!/bin/zsh
# Send one support ticket to the workaf-support trigger, the way a help desk's
# webhook would, and print the inbound event the control plane recorded.
# Usage: fire.sh ["ticket text"] [ticket-id]
set -euo pipefail
HERE=${0:A:h}; source $HERE/env.sh
TEXT=${1:-"Hi, the ledger CLI's total is wrong after I add a refund. I ran: ledger coffee 450 refund -500 and it printed \"2 entries, total 0.-50\". I expected -0.50."}
ID=${2:-T-$(date +%s)}
K=$(<$LOCAL/cp/secrets/cpkey)
TRIG=$(curl -fsS -H "X-API-Key: $K" http://127.0.0.1:$CP_PORT/api/v1/triggers \
  | jq -r '.triggers[] | select(.target_node_id=="workaf-support" and .target_reasoner=="handle_ticket") | .id' | head -1)
[[ -n $TRIG ]] || die "no workaf-support trigger; is the node up?"
BODY=$(jq -nc --arg id $ID --arg t $TEXT '{id:$id, subject:"ledger total wrong", ticket:$t}')
say "POST $CP_URL/sources/$TRIG  (ticket $ID)"
# The bearer is read from its file by curl's own config reader, never argv.
curl -fsS -X POST "http://127.0.0.1:$CP_PORT/sources/$TRIG" -H 'Content-Type: application/json' \
  -H "X-Ticket-Id: $ID" -K <(print -r -- "header = \"Authorization: Bearer $(<$LOCAL/cp/secrets/intake_bearer)\"") \
  -d "$BODY"
echo
