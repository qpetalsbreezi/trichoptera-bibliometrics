#!/usr/bin/env python3
"""Build docs/paper_preview.html from docs/paper.md with embedded figure PNGs."""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS = PROJECT_ROOT / "docs"
MD_PATH = DOCS / "paper.md"
OUT_PATH = DOCS / "paper_preview.html"

FIGURE_RE = re.compile(
    r"!\[(.*?)\]\(((?:\./)?figures/[^)]+)\)"
    r'|<img\s+[^>]*src="((?:\./)?figures/[^"]+)"[^>]*/?>',
    re.I,
)


def _embed_figures(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        path = (match.group(2) or match.group(3) or "").lstrip("./")
        alt = match.group(1) or Path(path).stem.replace("_", " ")
        img_path = DOCS / path
        if not img_path.is_file():
            return f"<p><em>Missing figure: {html.escape(path)}</em></p>"
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        alt_esc = html.escape(alt)
        return (
            f'<figure class="paper-figure">'
            f'<img src="data:image/png;base64,{b64}" alt="{alt_esc}" />'
            f"</figure>"
        )

    return FIGURE_RE.sub(repl, text)


def _md_to_html(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    in_table = False

    def inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
        return s

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            if in_table:
                out.append("</tbody></table>")
                in_table = False
            out.append("")
            i += 1
            continue

        if stripped.startswith("# "):
            if in_table:
                out.append("</tbody></table>")
                in_table = False
            out.append(f"<h1>{inline(stripped[2:])}</h1>")
            i += 1
            continue
        if stripped.startswith("## "):
            if in_table:
                out.append("</tbody></table>")
                in_table = False
            out.append(f"<h2>{inline(stripped[3:])}</h2>")
            i += 1
            continue
        if stripped.startswith("### "):
            if in_table:
                out.append("</tbody></table>")
                in_table = False
            out.append(f"<h3>{inline(stripped[4:])}</h3>")
            i += 1
            continue

        if stripped.startswith("|"):
            if not in_table:
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                out.append('<table class="paper-table"><thead><tr>')
                for c in cells:
                    out.append(f"<th>{inline(c)}</th>")
                out.append("</tr></thead><tbody>")
                in_table = True
                i += 1
                if i < len(lines) and lines[i].strip().startswith("|"):
                    i += 1  # skip separator row
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            out.append("<tr>")
            for c in cells:
                out.append(f"<td>{inline(c)}</td>")
            out.append("</tr>")
            i += 1
            continue

        if in_table:
            out.append("</tbody></table>")
            in_table = False

        if FIGURE_RE.search(stripped) or stripped.startswith("<p><img"):
            block = stripped
            if stripped.startswith("<p><img") and not stripped.endswith("</p>"):
                block = stripped
            elif stripped.startswith("<p><img"):
                pass
            out.append(_embed_figures(block))
            i += 1
            continue

        out.append(f"<p>{inline(stripped)}</p>")
        i += 1

    if in_table:
        out.append("</tbody></table>")

    return "\n".join(out)


def build() -> Path:
    md = MD_PATH.read_text(encoding="utf-8")
    body = _md_to_html(md)
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Paper preview</title>
  <style>
    body {{
      max-width: 920px;
      margin: 2rem auto;
      padding: 0 1.25rem 3rem;
      font-family: Georgia, "Times New Roman", serif;
      line-height: 1.55;
      color: #111;
    }}
    h1 {{ font-size: 1.55rem; line-height: 1.25; }}
    h2 {{ font-size: 1.25rem; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.25rem; }}
    h3 {{ font-size: 1.05rem; margin-top: 1.5rem; }}
    table.paper-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 1rem 0;
      font-size: 0.92rem;
    }}
    table.paper-table th, table.paper-table td {{
      border: 1px solid #ccc;
      padding: 0.35rem 0.5rem;
      vertical-align: top;
    }}
    table.paper-table th {{ background: #f5f5f5; }}
    .paper-figure {{ margin: 1.25rem 0; text-align: center; }}
    .paper-figure img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    p {{ margin: 0.75rem 0; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""
    OUT_PATH.write_text(page, encoding="utf-8")
    return OUT_PATH


if __name__ == "__main__":
    path = build()
    print(f"Wrote: {path.relative_to(PROJECT_ROOT)}")
