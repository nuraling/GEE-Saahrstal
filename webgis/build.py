#!/usr/bin/env python3
"""Build the password-protected WebGIS.

  SITE_USER=... SITE_PASS=... python build.py <data_dir> <site_dir>

<data_dir> is what tools/export_webgis.py wrote. Every data file is encrypted
with AES-256-GCM; the key is PBKDF2-SHA256(username:password, random salt,
250 000 rounds). The page holds only the salt, so the data cannot be read
without the credentials, even from a public repository. The credentials are
never written anywhere.
"""
import base64
import json
import os
import pathlib
import re
import sys

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

ITER = 250_000
here = pathlib.Path(__file__).parent
data_dir, site = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
user, pw = os.environ["SITE_USER"], os.environ["SITE_PASS"]
salt = os.urandom(16)
key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITER).derive(f"{user.strip()}:{pw}".encode())
aes = AESGCM(key)
(site / "data").mkdir(parents=True, exist_ok=True)
for old in (site / "data").glob("*"):
    old.unlink()


def put(name, raw: bytes):
    iv = os.urandom(12)
    ct = aes.encrypt(iv, raw, None)
    (site / "data" / f"{name}.enc.json").write_text(json.dumps({"iv": base64.b64encode(iv).decode(), "ct": base64.b64encode(ct).decode()}))


for f in sorted(data_dir.iterdir()):
    if f.name == "dsm_3m.json":                       # the elevation grid travels as raw uint16 bytes
        put("dsm_3m.bin", base64.b64decode(json.loads(f.read_text())["b64"]))
    elif f.suffix in (".json", ".geojson", ".webp"):
        put(f.name, f.read_bytes())
css = re.sub(r"url\(images/[^)]*\)", "none", (here / "leaflet-1.9.4.min.css").read_text())
html = (here / "index.src.html").read_text().replace("/*LEAFLET_CSS*/", css)
html = html.replace("/*SALT*/", base64.b64encode(salt).decode()).replace("/*ITER*/", str(ITER))
(site / "index.html").write_text(html)
(site / ".nojekyll").write_text("")
print(site / "index.html", sum(p.stat().st_size for p in (site / "data").iterdir()) // 1_000_000, "MB encrypted data")
