"""`ur install`: put the ur skill where the agents on this machine look for skills.

The skill is copied to one home (~/.config/agents/skills/ur unless UR_SKILLS_HOME says
otherwise) and linked from every other skills directory found under ~/.*/ and ~/.config/*/.
"""

from __future__ import annotations

import os
import shutil
from importlib import resources
from pathlib import Path

from .util import UrError

SKIP = {".cache", ".local", ".npm", ".cargo", ".rustup", ".git"}


def skill_source() -> Path:
    return Path(str(resources.files("ur") / "skill"))


def skills_home() -> Path:
    return Path(os.environ.get("UR_SKILLS_HOME") or Path.home() / ".config" / "agents" / "skills").expanduser()


def detect_skill_dirs(home: Path) -> list[Path]:
    found = []
    for pattern in (".*/skills", ".config/*/skills"):
        for path in sorted(home.glob(pattern)):
            if path.relative_to(home).parts[0] in SKIP:
                continue
            if path.is_dir():
                found.append(path)
    return found


def _is_ours(path: Path) -> bool:
    skill = path / "SKILL.md"
    return skill.is_file() and "\nname: ur\n" in skill.read_text()[:200]


def install_cmd(dirs: list[str], dry_run: bool, uninstall: bool) -> int:
    home_dir = skills_home()
    target = home_dir / "ur"
    others = [Path(d).expanduser() for d in dirs] if dirs else detect_skill_dirs(Path.home())
    others = [d for d in others if d.resolve() != home_dir.resolve()]
    verb = "Would" if dry_run else ""

    if uninstall:
        for d in others:
            link = d / "ur"
            if link.is_symlink() and link.resolve() == target.resolve():
                if not dry_run:
                    link.unlink()
                print(f"{verb + ' remove' if verb else 'Removed'} {link}")
        if target.exists() and _is_ours(target):
            if not dry_run:
                shutil.rmtree(target)
            print(f"{verb + ' remove' if verb else 'Removed'} {target}")
        return 0

    if target.exists() and not target.is_symlink() and not _is_ours(target):
        raise UrError(f"{target} exists and isn't the ur skill, so it was left alone.")
    if not dry_run:
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(skill_source(), target, ignore=shutil.ignore_patterns("__pycache__"))
    print(f"{verb + ' install' if verb else 'Installed'} the skill at {target}")
    for d in others:
        link = d / "ur"
        if link.is_symlink():
            if link.resolve() == target.resolve():
                print(f"  {link} already links to it")
                continue
            if not dry_run:
                link.unlink()
        elif link.exists():
            if not _is_ours(link):
                print(f"  {link} exists and isn't the ur skill; skipped")
                continue
            if not dry_run:
                shutil.rmtree(link)
        if not dry_run:
            link.symlink_to(target)
        print(f"  {verb + ' link' if verb else 'linked'} {link}")
    return 0
