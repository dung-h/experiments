#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-"$ROOT/work"}"
mkdir -p "$WORK"

clone_at() {
  local name="$1"
  local url="$2"
  local sha="$3"
  local dest="$WORK/$name"

  if [[ -e "$dest" ]]; then
    if [[ ! -d "$dest/.git" ]]; then
      echo "Refusing to reuse non-Git path: $dest" >&2
      exit 2
    fi
    if [[ -n "$(git -C "$dest" status --porcelain)" ]]; then
      echo "Refusing to modify dirty checkout: $dest" >&2
      exit 2
    fi
  else
    git clone "$url" "$dest"
  fi

  git -C "$dest" fetch --tags origin
  git -C "$dest" checkout --detach "$sha"
  git -C "$dest" diff --quiet
  git -C "$dest" diff --cached --quiet
}

overlay() {
  local source="$1"
  local destination="$2"
  [[ -d "$source" ]] || return 0
  cp -a "$source/." "$destination/"
}

clone_at mali https://github.com/mooselab/Quantum-Execution-Time-Prediction.git 32c392a6ece276f1ff046d4e30052d0571ff6dc6
overlay "$ROOT/tracks/mali/overlay" "$WORK/mali"

clone_at qonductor https://github.com/manosgior/Qonductor-SC25.git 5d1ac8a90cd574a23e7544e1044681641354ff67
git -C "$WORK/qonductor" apply --check "$ROOT/tracks/qonductor/qonductor_local_replication.patch"
git -C "$WORK/qonductor" apply "$ROOT/tracks/qonductor/qonductor_local_replication.patch"

clone_at cdaa https://github.com/mtkgv/cdaa.git 3c09c4c4feed19aea7cf710d22144c36063f0ebd
clone_at qcre https://github.com/mtkgv/qcre.git b3505da6bfa6d1184c21eb8410708cb759eadc48
overlay "$ROOT/tracks/cdaa_qcre/cdaa_overlay" "$WORK/cdaa"
# The upstream repository ignores these archived instruction-duration snapshots.
overlay "$ROOT/tracks/cdaa_qcre/data/instruction_durations" "$WORK/cdaa/1_depth_runtime/data/instruction_durations"

mkdir -p "$WORK/cdaa_independent/0_compilation"
overlay "$ROOT/tracks/cdaa_qcre/independent" "$WORK/cdaa_independent/0_compilation"
mkdir -p "$WORK/cdaa_independent/0_compilation/qasm/original"
cp -a "$WORK/cdaa/0_compilation/qasm/original/." "$WORK/cdaa_independent/0_compilation/qasm/original/"

clone_at vqcsim https://github.com/Security-FIT/VQCSim.git 7fc274d448c5e67a7a4149088afa594203cd5707
clone_at zero_setup https://github.com/arulrhikm/mps-pps-zero-setup-benchmarks.git c4485f1c267410d291a717d05aa12d493559f337

echo "Upstreams and overlays materialized in: $WORK"
