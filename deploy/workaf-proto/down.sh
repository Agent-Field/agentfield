#!/bin/zsh
# Stop everything and remove everything this kit made, on all three boxes:
# processes, homes, binaries, secrets, logs, clones and the build cache.
# Leaves ~/.codeaf on mac-studio untouched and the kit's own sources in git.
set -uo pipefail
HERE=${0:A:h}; source $HERE/env.sh
$HERE/node.sh clean
$HERE/engine.sh clean
$HERE/cp.sh clean
$HERE/build.sh clean
rm -rf $LOCAL
say "removed $LOCAL on spark"
for h in $NODE_HOST $ENGINE_HOST; do
  say "$h leftovers: $(ssh -o BatchMode=yes $h "ls -d ~/$KIT 2>/dev/null; pgrep -fl '$KIT' 2>/dev/null" | tr '\n' ' ')"
done
say "spark leftovers: $(ls -d $LOCAL 2>/dev/null; pgrep -fl "$KIT/stage" 2>/dev/null | tr '\n' ' ')"
