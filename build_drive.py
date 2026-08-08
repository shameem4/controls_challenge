"""Inline drive_data.json into drive_template.html -> test_drive.html.

The template keeps a `/*__DATA__*/` placeholder so page and data stay separately editable; this
joins them into the single self-contained file the artifact needs, since the artifact CSP blocks
external requests and the data cannot be fetched at runtime.
"""
from pathlib import Path

tpl = Path('drive_template.html').read_text()
out = tpl.replace('/*__DATA__*/', Path('drive_data.json').read_text())
assert '/*__DATA__*/' not in out, 'data placeholder not substituted'
assert not any(ord(c) > 127 for c in out), 'non-ASCII would mangle without a charset declaration'
Path('test_drive.html').write_text(out)
print(f'wrote test_drive.html ({len(out) / 1024:.0f} KB)')
