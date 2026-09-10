#!/usr/bin/env python3
"""Liste les architectures GPU (SM) reellement embarquees dans des roues ou des
bibliotheques natives CUDA.

Chaque .so compile par nvcc contient un "fatbin" : un conteneur qui regroupe
plusieurs versions du meme code machine, une par architecture. Ce script ouvre
ce conteneur et lit l'etiquette de chaque piece.

Rappel de compatibilite : un cubin compile pour sm_XY tourne sur toute carte
sm_XZ avec Z >= Y a l'interieur de la meme generation majeure X. Un cubin sm_86
fonctionne donc sur une Ada sm_89 ; un cubin sm_120 (Blackwell) ne fonctionne
pas, et son PTX ne peut pas etre retro-compile vers une architecture plus
ancienne. C'est l'origine du message "no kernel image is available".

Usage:
    python check_wheel_arch.py wheels/Linux/Torch2110/*.whl
    python check_wheel_arch.py --installed          # scanne le venv courant
    python check_wheel_arch.py --require 8.9 ...    # code retour 1 si incompatible
"""
import argparse
import collections
import os
import re
import struct
import sys
import zipfile

FATBIN_MAGIC = b"\x50\xed\x55\xba"  # 0xBA55ED50, little endian
KINDS = {1: "ptx", 2: "cubin"}


def scan_blob(data):
    """Retourne un Counter {('cubin', 86): n, ...} pour un binaire natif."""
    found = collections.Counter()
    for match in re.finditer(re.escape(FATBIN_MAGIC), data):
        base = match.start()
        try:
            _version, header_size = struct.unpack_from("<HH", data, base + 4)
            fat_size = struct.unpack_from("<Q", data, base + 8)[0]
        except struct.error:
            continue
        if header_size != 16 or not 0 < fat_size <= len(data):
            continue
        pos, end = base + header_size, base + header_size + fat_size
        while pos < end - 0x30:
            kind, _kver, entry_header = struct.unpack_from("<HHI", data, pos)
            if not 0x30 <= entry_header <= 0x200:
                break
            padded_payload = struct.unpack_from("<Q", data, pos + 8)[0]
            sm = struct.unpack_from("<I", data, pos + 0x1C)[0]
            found[(KINDS.get(kind, "kind%d" % kind), sm)] += 1
            pos += entry_header + padded_payload
    return found


def scan_wheel(path):
    found = collections.Counter()
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith(".so") or ".so." in name:
                found += scan_blob(archive.read(name))
    return found


def scan_target(path):
    """Un .whl, un repertoire de paquet, ou une bibliotheque native isolee."""
    if os.path.isdir(path):
        return scan_directory(path)
    if path.endswith(".whl"):
        return scan_wheel(path)
    with open(path, "rb") as handle:
        return scan_blob(handle.read())


def scan_directory(path):
    found = collections.Counter()
    for root, _dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".so") or ".so." in name:
                with open(os.path.join(root, name), "rb") as handle:
                    found += scan_blob(handle.read())
    return found


def runs_on(sm_list, target):
    """target = (major, minor). Vrai si un cubin de la liste est utilisable."""
    major, minor = target
    return any(
        sm // 10 == major and sm % 10 <= minor
        for kind, sm in sm_list
        if kind == "cubin"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", help="roues .whl ou repertoires")
    parser.add_argument("--installed", action="store_true",
                        help="scanne les extensions Trellis2 installees dans le venv courant")
    parser.add_argument("--require", metavar="X.Y",
                        help="echoue si une cible n'a aucun cubin utilisable sur cette capacite")
    args = parser.parse_args()

    targets = [(os.path.basename(t.rstrip("/")).split("-")[0] or t, t) for t in args.targets]
    if args.installed:
        import glob
        import importlib.util
        import sysconfig

        site_packages = sysconfig.get_paths()["purelib"]
        for module in ("cumesh", "flex_gemm", "o_voxel", "nvdiffrast", "nvdiffrec_render"):
            spec = importlib.util.find_spec(module)
            if spec is None:
                print("%-28s ABSENT du venv" % module)
                continue
            if spec.origin and os.path.isdir(os.path.dirname(spec.origin)):
                targets.append((module, os.path.dirname(spec.origin)))
            # Certains paquets (nvdiffrast) posent leur binaire a la racine de
            # site-packages, hors du repertoire du module : sans cela on
            # conclurait a tort a l'absence de code machine.
            for lib in glob.glob(os.path.join(site_packages, "*%s*.so" % module)):
                targets.append((module + " (natif)", lib))
    if not targets:
        parser.error("rien a scanner")

    required = None
    if args.require:
        major, minor = args.require.split(".")
        required = (int(major), int(minor))

    failures = 0
    for label, target in targets:
        found = scan_target(target)
        if not found:
            print("%-28s aucun code machine ici (sources compilees a la volee ?)" % label)
            continue
        summary = ", ".join(
            "%s sm_%d" % (kind, sm) for kind, sm in sorted(found, key=lambda k: (k[0], k[1]))
        )
        verdict = ""
        if required:
            ok = runs_on(found, required)
            verdict = "  [%s sur sm_%d%d]" % ("OK" if ok else "INCOMPATIBLE",
                                              required[0], required[1])
            failures += 0 if ok else 1
        print("%-28s %s%s" % (label, summary, verdict))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
