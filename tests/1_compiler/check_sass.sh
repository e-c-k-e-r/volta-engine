#!/usr/bin/env bash
set -euo pipefail

# Tier 1 SASS Invariant Audit for Volta SM70
TARGET="${1:-build/libengine_kernels.a}"

if [ ! -f "$TARGET" ]; then
	echo "[SKIP] Target $TARGET does not exist yet."
	exit 0
fi

echo "[*] Auditing SASS Tensor Core (HMMA) instructions in $TARGET..."
HMMA_COUNT=$(cuobjdump -sass "$TARGET" 2>/dev/null | grep -c "HMMA" || true)

echo "[+] SASS HMMA instruction count: $HMMA_COUNT"
if [ "$HMMA_COUNT" -lt 30000 ]; then
	echo "[-] FATAL: Native Volta tensor cores (HMMA) were stripped or missing! Count: $HMMA_COUNT (expected >= 30000)"
	exit 1
fi

echo "[+] Tier 1 SASS Invariant PASS."
