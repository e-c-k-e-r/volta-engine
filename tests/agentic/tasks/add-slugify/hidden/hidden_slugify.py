from textutil import slugify
cases = {"Hello, World!": "hello-world", "  A  B  ": "a-b", "x--y__z": "x-y-z", "": "", "123 abc": "123-abc", "---": ""}
for src, want in cases.items():
    got = slugify(src)
    assert got == want, (src, got, want)
