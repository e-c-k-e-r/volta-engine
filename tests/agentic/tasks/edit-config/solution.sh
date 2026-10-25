#!/bin/bash
# reference solution, applied in the work directory (self-test only)
python3 - <<'EOF'
import json
c = json.load(open("config.json"))
c["server"]["port"] = 9090
c["server"]["tls"] = True
c["features"].append("metrics")
json.dump(c, open("config.json", "w"), indent=2)
EOF
