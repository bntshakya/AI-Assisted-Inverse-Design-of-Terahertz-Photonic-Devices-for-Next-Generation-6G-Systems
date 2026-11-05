"""Pack the study website into ONE offline HTML file (models, data and images inside).

Double-click the result to open it in any browser: no server, no internet, no claude.ai account needed
(fonts fall back to the system fonts when offline).

Run from the Project folder:  python learn/make_offline_site.py
Reads learn/site/ (index.html, web_data.json, *.png); writes docs/Valley_Crystal_Study_Lab.html
"""
import base64
import os
import re
import sys

SITE = "learn/site"
OUT = "docs/Valley_Crystal_Study_Lab.html"
FETCH = re.compile(r'fetch\("web_data\.json"\)\.then\(r=>r\.json\(\)\)\.then\(d=>\{(.*?)\}\)\.catch\(e=>\{(.*?)\}\);')
HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>
"""


def main():
    html = open(os.path.join(SITE, "index.html"), encoding="utf-8").read()
    data = open(os.path.join(SITE, "web_data.json"), encoding="utf-8").read()
    if "</script" in data:
        sys.exit("web_data.json contains '</script', cannot embed safely")
    m = FETCH.search(html)
    if not m:
        sys.exit("could not find the fetch(...) line in index.html: update make_offline_site.py")
    body = m.group(1).replace("D=d;", 'D=JSON.parse(document.getElementById("webdata").textContent);', 1)
    loader = "try{" + body + "}catch(e){" + m.group(2) + "}"
    html = html[:m.start()] + loader + html[m.end():]
    for png in re.findall(r'src="([\w.-]+\.png)"', html):
        b64 = base64.b64encode(open(os.path.join(SITE, png), "rb").read()).decode()
        html = html.replace(f'src="{png}"', f'src="data:image/png;base64,{b64}"')
    i = html.index('<div class="app">')
    head, body = html[:i], html[i:]
    j = body.rindex("<script>")
    body = body[:j] + f'<script type="application/json" id="webdata">{data}</script>\n' + body[j:]
    out = HEAD + head + "</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
