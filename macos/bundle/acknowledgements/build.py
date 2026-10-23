"""Write Acknowledgements.json: every component Whiz.app ships, with its licence.

Run by build-engine.sh with the BUNDLED Python (-I), so the Python packages
listed are the ones installed in the engine and the versions shown are the
versions shipped. See manifest.json for the format of the curated part.

    python3.13 -I -B build.py --out Acknowledgements.json \
        --file whisper.cpp=<src>/LICENSE --version whisper.cpp=1.9.5 \
        --file sparkle=<Sparkle>/LICENSE --resolved Package.resolved \
        --onnxruntime <site-packages>/sherpa_onnx/lib/libonnxruntime.dylib

Fails loudly rather than ship an incomplete list: a Python package without a
licence file, a missing file, or a component whose version moved away from
the licence text copied for it ('expect').
"""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import sys
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Ours, or not shipped (build-engine.sh removes pip from the bundle).
SKIP_PACKAGES = {"whiz", "pip"}

LICENSE_FILE = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE|AUTHORS)", re.IGNORECASE)


def fail(message: str) -> None:
    sys.exit(f"acknowledgements: {message}")


def onnxruntime_version(dylib: str) -> str:
    """Ask the library itself (OrtGetApiBase()->GetVersionString())."""

    class OrtApiBase(ctypes.Structure):
        _fields_ = [("GetApi", ctypes.c_void_p),
                    ("GetVersionString", ctypes.CFUNCTYPE(ctypes.c_char_p))]

    lib = ctypes.CDLL(dylib)
    lib.OrtGetApiBase.restype = ctypes.POINTER(OrtApiBase)
    return lib.OrtGetApiBase().contents.GetVersionString().decode()


def builtin_versions() -> dict[str, str]:
    """Versions of what is linked into the running (bundled) Python."""
    import decimal
    import platform
    import pyexpat
    import sqlite3
    import ssl

    return {
        "python": platform.python_version(),
        "openssl": ssl.OPENSSL_VERSION.split()[1],
        "sqlite": sqlite3.sqlite_version,
        "expat": pyexpat.EXPAT_VERSION.removeprefix("expat_"),
        "mpdecimal": decimal.__libmpdec_version__,
    }


def short_license(meta: metadata.PackageMetadata) -> str:
    expression = meta.get("License-Expression")
    if expression:
        return expression
    legacy = (meta.get("License") or "").strip()
    if legacy and "\n" not in legacy and len(legacy) <= 40:
        return legacy
    for classifier in meta.get_all("Classifier") or []:
        if classifier.startswith("License :: OSI Approved :: "):
            return classifier.rsplit(" :: ", 1)[1].removesuffix(" License")
    return ""


def home_page(meta: metadata.PackageMetadata) -> str | None:
    urls = {}
    for entry in meta.get_all("Project-URL") or []:
        label, _, url = entry.partition(",")
        urls[label.strip().lower()] = url.strip()
    for label in ("homepage", "home", "source", "source code", "repository", "github"):
        if label in urls:
            return urls[label]
    return meta.get("Home-page") or None


def licence_texts(dist: metadata.Distribution) -> list[tuple[str, str]]:
    texts = []
    for file in dist.files or []:
        parts = Path(str(file)).parts
        in_info = parts and parts[0].endswith(".dist-info")
        if in_info and ("licenses" in parts or LICENSE_FILE.match(parts[-1])):
            texts.append((parts[-1], Path(dist.locate_file(file)).read_text(encoding="utf-8", errors="replace")))
    return sorted(texts)


def sniff_license(text: str) -> str:
    """For a package whose metadata names no licence: read it off the text."""
    if "Apache License" in text and "Version 2.0" in text:
        return "Apache-2.0"
    if "Permission is hereby granted, free of charge" in text:
        return "MIT"
    return "See licence"


def python_packages(summaries: dict[str, str]) -> list[dict]:
    dists = {}
    for dist in metadata.distributions():
        key = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()   # PEP 503
        if key not in SKIP_PACKAGES:
            dists[key] = dist
    entries: dict[str, dict] = {}
    unlicensed = []
    for key, dist in sorted(dists.items()):
        texts = licence_texts(dist)
        if not texts:
            unlicensed.append(key)
            continue
        meta = dist.metadata
        text = join(texts)
        entries[key] = {
            "name": meta["Name"].replace("_", "-"),
            "version": dist.version,
            "license": short_license(meta) or sniff_license(text),
            "url": home_page(meta),
            "summary": summaries.get(key) or meta.get("Summary") or "",
            "text": text,
        }
    # A split wheel (sherpa-onnx-core beside sherpa-onnx) carries its
    # licence in the main package; anything else unlicensed is an error.
    for key in unlicensed:
        parent = next((p for p in entries if key.startswith(p + "-")), None)
        if parent is None:
            fail(f"the Python package {key} has no licence file; add it to manifest.json by hand")
        entries[parent]["summary"] += f" Includes {dists[key].metadata['Name']}."
    return sorted(entries.values(), key=lambda e: e["name"].lower())


def join(texts: list[tuple[str, str]]) -> str:
    if len(texts) == 1:
        return texts[0][1].strip() + "\n"
    return "\n\n".join(f"── {name} ──\n\n{text.strip()}" for name, text in texts) + "\n"


def curated(component: dict, files: dict[str, Path], versions: dict[str, str]) -> dict:
    entry = {key: component[key] for key in ("name", "license", "url", "summary") if key in component}
    version = component.get("version")
    if version:
        if version.startswith("@"):
            version = versions.get(version[1:]) or fail(f"no version given for {version}")
        entry["version"] = version
    expected = component.get("expect")
    if expected and version != expected:
        fail(f"{component['name']} is now {version}, but its licence text was copied for "
             f"{expected}: refresh it in bundle/acknowledgements, then 'expect'")
    texts = []
    for name in component.get("files", []):
        path = files.get(name[1:]) if name.startswith("@") else HERE / name
        if path is None or not path.is_file():
            fail(f"licence file {name} for {component['name']} is missing")
        texts.append((path.name, path.read_text(encoding="utf-8", errors="replace")))
    if texts:
        entry["text"] = join(texts)
    return entry


def pairs(values: list[str], flag: str) -> dict[str, str]:
    result = {}
    for value in values:
        name, sep, rest = value.partition("=")
        if not sep:
            fail(f"{flag} expects name=value, got {value!r}")
        result[name] = rest
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--file", action="append", default=[])
    parser.add_argument("--version", action="append", default=[])
    parser.add_argument("--resolved", help="Package.resolved: each Swift package's version, by identity")
    parser.add_argument("--onnxruntime", required=True, help="path to libonnxruntime.dylib")
    args = parser.parse_args()

    files = {name: Path(path) for name, path in pairs(args.file, "--file").items()}
    versions = builtin_versions() | pairs(args.version, "--version")
    versions["onnxruntime"] = onnxruntime_version(args.onnxruntime)
    if args.resolved:
        for pin in json.loads(Path(args.resolved).read_text())["pins"]:
            versions[pin["identity"]] = pin["state"]["version"]

    manifest = json.loads((HERE / "manifest.json").read_text())
    groups = []
    for group in manifest["groups"]:
        components = []
        for component in group["components"]:
            if "python_packages" in component:
                components.extend(python_packages(component["python_packages"]))
            else:
                components.append(curated(component, files, versions))
        groups.append({"title": group["title"], "components": components})

    Path(args.out).write_text(json.dumps({"groups": groups}, indent=1, ensure_ascii=False) + "\n")
    count = sum(len(g["components"]) for g in groups)
    print(f"Acknowledgements: {count} components → {args.out}")


if __name__ == "__main__":
    main()
