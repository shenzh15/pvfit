"""Check relative Markdown file links in the README and documentation."""

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
errors = []
for source in [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]:
    text = re.sub(r"```.*?```", "", source.read_text(), flags=re.S)
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
        target = target.split()[0].strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        if not (source.parent / unquote(parsed.path)).exists():
            errors.append(f"{source.relative_to(ROOT)}: {target}")
if errors:
    raise SystemExit("Broken local links:\n" + "\n".join(errors))
print("Local documentation file links resolve.")
