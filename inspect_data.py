import json
import glob
from pathlib import Path

triggers = glob.glob('expanded/triggers/*.json')
print(f"Total triggers: {len(triggers)}")
for f in triggers[:20]:
    d = json.load(open(f))
    print(f"{d.get('id')}: kind={d.get('kind')}, scope={d.get('scope')}, payload_keys={list(d.get('payload', {}).keys())}")
