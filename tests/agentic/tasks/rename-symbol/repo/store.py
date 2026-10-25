def fetch_data(path):
    """fetch_data reads one JSON record per line."""
    import json
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]
