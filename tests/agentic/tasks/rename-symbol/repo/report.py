from store import fetch_data


def total(path):
    return sum(r["n"] for r in fetch_data(path))
