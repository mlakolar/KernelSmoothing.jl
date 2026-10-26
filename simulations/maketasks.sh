#!/bin/bash
# Writes the task list of a simulation campaign, one line "STUDY DESIGN N FIRSTREP
# NREPS" per task, to tasks.txt in the given directory, for the given studies (by
# default all), and a short list for a canary run to tasks-quick.txt.
# Usage: maketasks.sh [OUTDIR [STUDY ...]]
set -euo pipefail
out=${1:-.}
shift || true
studies=" ${*:-locallik quantreg shape bounded rdlocrand regression density cluster honest rd rddensity lpdboundary mvkde partiallylinear} "
: > "$out/tasks.txt"
: > "$out/tasks-quick.txt"
add() {   # study, designs, sample sizes, replications, replications per task
  local study=$1 designs=$2 ns=$3 reps=$4 per=$5
  [[ $studies == *" $study "* ]] || return 0
  for d in $designs; do for n in $ns; do
    for ((first = 1; first <= reps; first += per)); do echo "$study $d $n $first $per" >> "$out/tasks.txt"; done
  done; done
  echo "$study ${designs%% *} ${ns%% *} 1 2" >> "$out/tasks-quick.txt"
}
add locallik "logit-sine poisson-sine gamma-sine negbin-sine" "500 2000" 2000 100
add quantreg "sine-het-0.25 sine-het-0.5 sine-het-0.75 sine-t3-0.25 sine-t3-0.5 sine-t3-0.75" "500 2000" 2000 250
add shape "logistic flat sine" "200 1000" 2000 250
add bounded "beta13 beta22 uniform" "200 1000" 1000 100
add rdlocrand "null-flat null-slope alt-flat alt-slope" "500 2000" 2000 250
add regression "sine sine-heteroskedastic peak normal-sine t3-sine" "500 2000 10000" 2000 100
add density "normal bimodal t5" "500 2000 10000" 2000 200
add cluster "balanced-20 balanced-50 balanced-200 unbalanced-20 unbalanced-50 unbalanced-200" "1" 2000 200
add honest "sine peak" "500 2000" 2000 250
add rd "lee ludwig-miller" "500 5000" 2000 250
add rddensity "normal-0 normal-0.1 normal-0.25 normal-0.5 beta-0 beta-0.1 beta-0.25 beta-0.5" "1000 5000" 2000 250
add lpdboundary "exponential beta13" "500 2000 10000" 2000 250
add mvkde "correlated bimodal skewed" "200" 500 25
add mvkde "correlated bimodal skewed" "1000" 200 10
add partiallylinear "sine sine-heteroskedastic logistic" "200 1000" 2000 500
wc -l "$out/tasks.txt" "$out/tasks-quick.txt"
