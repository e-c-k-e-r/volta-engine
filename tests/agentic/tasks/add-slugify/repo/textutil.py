def title(text):
    return " ".join(w.capitalize() for w in text.split())


def truncate(text, n):
    return text if len(text) <= n else text[: n - 1] + "…"
