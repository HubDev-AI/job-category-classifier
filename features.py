import re


def tokenize(text):
    words = re.findall(r'[a-z0-9+#.]+', text.lower())
    return [w.rstrip('.') for w in words if w.rstrip('.')]
