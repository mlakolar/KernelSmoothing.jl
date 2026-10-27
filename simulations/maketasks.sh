#!/bin/bash
# Writes the task list of a simulation campaign, one line "STUDY DESIGN N FIRSTREP
# NREPS" per task, to tasks.txt in the given directory, for the given studies (by
# default all), and a short list for a canary run to tasks-quick.txt.
# Usage: maketasks.sh [OUTDIR [STUDY ...]]
# PARTIALLYLINEAR_CONFIRMATION=1 also writes tasks-partiallylinear-confirmation.txt
# for replications 2001–4000 when partiallylinear is selected; default lists stay the same.
set -euo pipefail
out=${1:-.}
shift || true
studies=" ${*:-locallik quantreg shape bounded rdlocrand regression density cluster honest rd rddensity lpdboundary mvkde partiallylinear mvbinning} "
: > "$out/tasks.txt"
: > "$out/tasks-quick.txt"
add() {   # study, designs, sample sizes, replications, replications per task, optional canary
  local study=$1 designs=$2 ns=$3 reps=$4 per=$5
  [[ $studies == *" $study "* ]] || return 0
  for d in $designs; do for n in $ns; do
    for ((first = 1; first <= reps; first += per)); do echo "$study $d $n $first $per" >> "$out/tasks.txt"; done
  done; done
  if [[ ${6:-yes} == yes ]]; then
    echo "$study ${designs%% *} ${ns%% *} 1 2" >> "$out/tasks-quick.txt"
  fi
}
add locallik "logit-sine poisson-sine gamma-sine negbin-sine" "500 2000" 2000 100
add quantreg "sine-het-0.25 sine-het-0.5 sine-het-0.75 sine-t3-0.25 sine-t3-0.5 sine-t3-0.75 qy2-0.2 qy2-0.5 qy2-0.8" "500 2000" 2000 250
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
partiallylinear_designs="sine sine-heteroskedastic logistic quadratic cancellation g-zero linear-means"
# At 0.1–0.2 s per replication at n = 1000, 1000 replications take a few minutes.
add partiallylinear "$partiallylinear_designs" "200 1000" 2000 1000 no
if [[ $studies == *" partiallylinear "* ]]; then
  echo "partiallylinear sine 1000 1 2" >> "$out/tasks-quick.txt"
  if [[ ${PARTIALLYLINEAR_CONFIRMATION:-0} == 1 ]]; then
    : > "$out/tasks-partiallylinear-confirmation.txt"
    for d in $partiallylinear_designs; do for n in 200 1000; do
      for first in 2001 3001; do
        echo "partiallylinear $d $n $first 1000" >> "$out/tasks-partiallylinear-confirmation.txt"
      done
    done; done
  fi
fi
# Exact 2-D SCV took 106 s at n = 10⁴ (notes/timings.md). Allow one replication
# per task there and above; 3-D has no measured baseline. Check its canary before
# running the full list. Two replications at 10³ assume roughly quadratic scaling.
add mvbinning "correlated bimodal skewed trivariate" "1000" 100 2 no
add mvbinning "correlated bimodal skewed trivariate" "10000" 50 1 no
add mvbinning "correlated bimodal skewed trivariate" "100000" 10 1 no
add mvbinning "bimodal-g51 bimodal-g101 bimodal-g151 bimodal-g301 trivariate-g21 trivariate-g31 trivariate-g51" "10000" 50 1 no
if [[ $studies == *" mvbinning "* ]]; then
  # Match the full task size: the resume key uses FIRSTREP but not NREPS.
  echo "mvbinning bimodal 1000 1 2" >> "$out/tasks-quick.txt"
  echo "mvbinning trivariate 1000 1 2" >> "$out/tasks-quick.txt"
fi
wc -l "$out/tasks.txt" "$out/tasks-quick.txt"
if [[ $studies == *" partiallylinear "* && ${PARTIALLYLINEAR_CONFIRMATION:-0} == 1 ]]; then
  wc -l "$out/tasks-partiallylinear-confirmation.txt"
fi
