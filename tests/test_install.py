from pathlib import Path


def test_install_links_every_skills_dir(tmp_path, ur):
    home = Path(ur.env["HOME"])
    shared = home / ".config" / "agents" / "skills"
    shared.mkdir(parents=True)
    (home / ".agents").mkdir()
    (home / ".agents" / "skills").symlink_to(shared)  # already the shared home
    (home / ".tool-a" / "skills").mkdir(parents=True)
    (home / ".config" / "tool-b" / "skills").mkdir(parents=True)
    (home / ".cache" / "skills").mkdir(parents=True)  # not a skills home
    foreign = home / ".tool-c" / "skills" / "ur"
    foreign.mkdir(parents=True)
    (foreign / "SKILL.md").write_text("---\nname: something-else\n---\n")

    out = ur("install", cwd=tmp_path).stdout
    skill = shared / "ur" / "SKILL.md"
    assert skill.read_text().startswith("---\nname: ur\n")
    assert "Commands" in skill.read_text()
    for d in (home / ".tool-a" / "skills", home / ".config" / "tool-b" / "skills"):
        assert (d / "ur").is_symlink() and (d / "ur").resolve() == (shared / "ur").resolve()
    assert not (home / ".cache" / "skills" / "ur").exists()
    assert not (home / ".agents" / "skills" / "ur").is_symlink()  # it's the shared dir itself
    assert "isn't the ur skill; skipped" in out
    assert (foreign / "SKILL.md").read_text().startswith("---\nname: something-else")

    # Installing again is a no-op for the links; uninstalling removes only what ur made.
    assert "already links to it" in ur("install", cwd=tmp_path).stdout
    ur("install", "--uninstall", cwd=tmp_path)
    assert not (shared / "ur").exists()
    assert not (home / ".tool-a" / "skills" / "ur").exists()
    assert foreign.exists()


def test_skill_names_no_products():
    text = (Path(__file__).resolve().parents[1] / "src" / "ur" / "skill" / "SKILL.md").read_text().lower()
    for name in ("claude", "codex", "opencode", "cursor", "gemini", "openai", "anthropic", "t3", "copilot"):
        assert name not in text, name
