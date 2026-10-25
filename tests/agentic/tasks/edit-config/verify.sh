#!/bin/bash
# runs in the task's work directory after the run; exit 0 = pass
python3 - <<'EOF' || exit 1
import json
c = json.load(open("config.json"))
assert c["server"] == {"host": "0.0.0.0", "port": 9090, "workers": 4, "tls": True}, c["server"]
assert c["features"] == ["auth", "cache", "metrics"], c["features"]
assert c["name"] == "demo" and c["log"] == {"level": "info"}
EOF
