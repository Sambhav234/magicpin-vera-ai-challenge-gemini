import json

test_pairs = json.load(open('expanded/test_pairs.json'))
print(f"Type: {type(test_pairs)}, keys: {list(test_pairs.keys())}")
if isinstance(test_pairs, dict):
    first_key = list(test_pairs.keys())[0]
    print(f"Sample under '{first_key}': {test_pairs[first_key]}")
