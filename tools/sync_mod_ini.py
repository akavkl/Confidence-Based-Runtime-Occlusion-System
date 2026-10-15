# Post-build: adds the settings res/CBRO.ini documents that the MO2 mod folder's CBRO.ini lacks, each with its comment
# block, after the setting that precedes it in res/CBRO.ini (or at its section's start). Nothing else in the user's file
# changes: their values, comments and order stay. A missing file is copied whole.
# Usage: sync_mod_ini.py <res/CBRO.ini> <mod folder>/F4SE/Plugins/CBRO.ini
import os
import re
import sys


def entries(lines):
    """(section, key, first line of the comment block right above it, the key's line) in file order."""
    out, section, start = [], None, None
    for i, line in enumerate(lines):
        s = line.strip()
        if re.fullmatch(r"\[[^\]]+\]", s):
            section, start = s[1:-1], None
        elif s.startswith(";"):
            start = i if start is None else start
        elif "=" in s and section is not None:
            out.append((section, s.split("=", 1)[0].strip(), i if start is None else start, i))
            start = None
        else:
            start = None
    return out


def main(shipped_path, user_path):
    shipped = open(shipped_path, encoding="utf-8").read().replace("\r\n", "\n").split("\n")
    if not os.path.exists(user_path):
        open(user_path, "w", encoding="utf-8", newline="\n").write("\n".join(shipped))
        print(f"CBRO.ini: copied {shipped_path} (none in the mod folder)")
        return
    raw = open(user_path, "rb").read()
    eol = "\r\n" if b"\r\n" in raw else "\n"
    user = raw.decode("utf-8").replace("\r\n", "\n").split("\n")
    have = {(s, k) for s, k, _, _ in entries(user)}
    order = entries(shipped)
    added = []
    for index, (section, key, start, line) in enumerate(order):
        if (section, key) in have:
            continue
        anchor = None
        for s, k, _, _ in reversed(order[:index]):
            if s != section:
                break
            if (s, k) in have:
                anchor = (s, k)
                break
        if anchor:
            at = next(i for s, k, _, i in entries(user) if (s, k) == anchor) + 1
        else:
            headers = [i for i, l in enumerate(user) if l.strip() == f"[{section}]"]
            if not headers:
                user += ["", f"[{section}]"]
                headers = [len(user) - 1]
            at = headers[0] + 1
        user[at:at] = shipped[start:line + 1]
        have.add((section, key))
        added.append(f"[{section}] {shipped[line].strip()}")
    if added:
        open(user_path, "wb").write(eol.join(user).encode("utf-8"))
        print(f"CBRO.ini: added {len(added)} setting(s) the mod folder's file lacked: " + ", ".join(added))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
