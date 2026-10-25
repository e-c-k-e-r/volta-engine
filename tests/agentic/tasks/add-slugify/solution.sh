#!/bin/bash
# reference solution, applied in the work directory (self-test only)
cat >> textutil.py <<'EOF'


def slugify(text):
    import re
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
EOF
cat >> test_textutil.py <<'EOF'


class S(unittest.TestCase):
    def test_slugify(self):
        from textutil import slugify
        self.assertEqual(slugify("Hello, World!"), "hello-world")
EOF
