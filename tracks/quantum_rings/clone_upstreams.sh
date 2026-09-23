#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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

clone_at quantum_rings_challenge https://github.com/iQuHACK/2026-Quantum-Rings.git 1e2247f708136b2c6da99d7f98564ea2b15474f9
clone_at spirit_sprinters https://github.com/woody-hulse/quantum-rings.git 14b5ea14dea8f6b55edf632030f5c022e8ad7373
clone_at softlocked https://github.com/SoftLocked/2026-Quantum-Rings.git da13c074d68c2fe3a1053c751234ee0916d73739

echo "Quantum Rings third-party trees materialized in: $WORK"
echo "Do not treat these clones as this repository's scientific result."
echo "The independent audit is artifacts/quantum_rings/REPORT.md"
