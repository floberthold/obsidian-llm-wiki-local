"""
obsidian-llm-wiki CLI (olw)

Commands:
  init     — create vault structure (or adopt existing)
  ingest   — analyze raw notes
  compile  — synthesize notes into wiki articles (writes to .drafts/)
  approve  — publish drafts to wiki/
  reject   — discard a draft
  status   — show vault health and pending drafts
  undo     — revert last N [olw] git commits
  query    — RAG-powered Q&A (Phase 2)
  lint     — check wiki health (Phase 2)
  watch    — file watcher daemon (Phase 3)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt
from rich.table import Table

console = Console()
err_console = Console(stderr=True, style="bold red")


# ── Context helpers ───────────────────────────────────────────────────────────


def _load_config(vault_str: str | None, **kwargs):
    from .config import Config
    from .global_config import load_global_config

    if vault_str is None:
        gcfg = load_global_config()
        vault_str = gcfg.vault if gcfg and gcfg.vault else None

    if not vault_str:
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            if (parent / "wiki.toml").exists():
                vault_str = str(parent)
                break

    if not vault_str:
        click.echo(
            "Error: no vault specified. Use --vault, set OLW_VAULT, run `olw setup`, "
            "or cd into a vault directory.",
            err=True,
        )
        sys.exit(1)
    return Config.from_vault(Path(vault_str), **kwargs)


def _load_db(config):
    from .state import StateDB

    return StateDB(config.state_db_path)


def _load_deps(config):
    from .client_factory import LLMError, build_client

    client = build_client(config)
    try:
        client.require_healthy()
    except LLMError as e:
        err_console.print(str(e))
        sys.exit(1)
    db = _load_db(config)
    return client, db


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    whole = max(0, int(round(seconds)))
    mins, secs = divmod(whole, 60)
    return f"{mins:02d}:{secs:02d}"


# ── CLI root ──────────────────────────────────────────────────────────────────


@click.group()
@click.version_option(package_name="obsidian-llm-wiki")
def cli():
    """obsidian-llm-wiki (olw) — 100% local Obsidian → wiki pipeline.

    Run `olw setup` for interactive configuration.
    """
    import logging

    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=False)],
    )
    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


# ── init ──────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("vault_path", type=click.Path())
@click.option("--existing", is_flag=True, help="Adopt an existing Obsidian vault")
@click.option("--non-interactive", is_flag=True)
def init(vault_path: str, existing: bool, non_interactive: bool):
    """Create vault structure and initialise olw."""
    from .git_ops import git_init

    vault = Path(vault_path).expanduser().resolve()
    vault.mkdir(parents=True, exist_ok=True)

    if existing:
        _init_existing(vault, non_interactive)
    else:
        _init_fresh(vault)

    # Write or sync wiki.toml from global config
    toml_path = vault / "wiki.toml"
    from .config import default_wiki_toml
    from .global_config import load_global_config

    gcfg = load_global_config()
    provider_name = gcfg.provider_name if gcfg and gcfg.provider_name else "ollama"
    # Only fall back to Ollama-specific model names when using Ollama; cloud providers
    # must have been configured explicitly via `olw setup`.
    _ollama = provider_name == "ollama"
    fast = gcfg.fast_model if gcfg and gcfg.fast_model else ("qwen3:4b" if _ollama else "")
    heavy = gcfg.heavy_model if gcfg and gcfg.heavy_model else ("qwen3.6:35b-a3b" if _ollama else "")
    provider_url = gcfg.provider_url if gcfg and gcfg.provider_url else None
    ollama_url = gcfg.ollama_url if gcfg and gcfg.ollama_url else "http://localhost:11434"
    effective_url = provider_url or ollama_url
    azure_api_version = gcfg.azure_api_version if gcfg and gcfg.azure_api_version else None

    if not toml_path.exists():
        from .providers import get_provider

        prov_info = get_provider(provider_name)
        timeout = prov_info.default_timeout if prov_info else 600.0
        toml_path.write_text(
            default_wiki_toml(
                fast,
                heavy,
                ollama_url=ollama_url,
                provider_name=provider_name,
                provider_url=effective_url if provider_name != "ollama" else None,
                provider_timeout=timeout,
                azure_api_version=azure_api_version,
            )
        )
    else:
        # Existing vault: patch model/URL fields from global config so that
        # olw setup changes are reflected without overwriting pipeline settings.
        _sync_wiki_toml_models(
            toml_path,
            fast,
            heavy,
            effective_url,
            provider_name=provider_name if provider_name != "ollama" else None,
        )

    # Init git
    git_init(vault)

    # Create .gitignore
    gi = vault / ".gitignore"
    if not gi.exists():
        gi.write_text(".DS_Store\n.olw/chroma/\n.olw/state.db\n.obsidian/workspace.json\n*.log\n")

    console.print(f"[green]Vault initialised:[/green] {vault}")
    console.print("Next steps:")
    console.print("  1. Drop .md notes into [bold]raw/[/bold]")
    console.print("  2. Run [bold]olw run[/bold]  (ingest + compile + lint in one step)")
    console.print("  3. Review drafts: [bold]olw review[/bold]")


def _sync_wiki_toml_models(
    toml_path: Path,
    fast: str,
    heavy: str,
    ollama_url: str,
    provider_name: str | None = None,
) -> None:
    """Patch fast/heavy model, URL, and optionally provider name in an existing wiki.toml.

    Preserves all other settings (pipeline, rag, etc.) so user customisations
    are not lost. Only updates fields that come from global config.

    URL is only updated within the [ollama] or [provider] section, never globally,
    so switching providers cannot overwrite unrelated url= fields.
    """
    import re

    text = toml_path.read_text(encoding="utf-8")
    original = text

    def _replace_value(t: str, key: str, value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return re.sub(
            rf'^({re.escape(key)}\s*=\s*)".+"',
            rf'\g<1>"{escaped}"',
            t,
            flags=re.MULTILINE,
        )

    def _replace_in_section(t: str, section: str, key: str, value: str) -> str:
        """Replace key=value only within the named TOML section."""
        escaped_val = value.replace("\\", "\\\\").replace('"', '\\"')
        # Match the section header then capture everything until the next section or EOF
        pattern = rf'(\[{re.escape(section)}\][^\[]*?)^({re.escape(key)}\s*=\s*)".+"'
        replacement = rf'\1\2"{escaped_val}"'
        return re.sub(pattern, replacement, t, flags=re.MULTILINE | re.DOTALL)

    text = _replace_value(text, "fast", fast)
    text = _replace_value(text, "heavy", heavy)
    # Update URL only in [ollama] or [provider] sections to avoid clobbering other urls
    for section in ("ollama", "provider"):
        text = _replace_in_section(text, section, "url", ollama_url)
    if provider_name is not None:
        if "[provider]" not in text:
            console.print(
                f"  [yellow]Warning:[/yellow] wiki.toml has no [provider] section — "
                f"provider '{provider_name}' not applied. "
                f"Delete wiki.toml and re-run [bold]olw init[/bold] to regenerate it."
            )
        else:
            text = _replace_in_section(text, "provider", "name", provider_name)

    if text != original:
        toml_path.write_text(text, encoding="utf-8")
        console.print(f"[dim]wiki.toml updated: fast={fast}, heavy={heavy}, url={ollama_url}[/dim]")


def _init_fresh(vault: Path) -> None:
    for d in ["raw", "wiki", "wiki/.drafts", "wiki/sources", ".olw", ".olw/chroma"]:
        (vault / d).mkdir(parents=True, exist_ok=True)
    _write_vault_schema(vault)
    _write_index(vault)
    console.print("[dim]Created fresh vault structure[/dim]")


def _init_existing(vault: Path, non_interactive: bool) -> None:
    note_count = sum(1 for _ in vault.rglob("*.md"))
    console.print(f"Found [bold]{note_count}[/bold] existing .md files in {vault}")

    for d in ["raw", "wiki", "wiki/.drafts", "wiki/sources", ".olw", ".olw/chroma"]:
        (vault / d).mkdir(parents=True, exist_ok=True)

    if not non_interactive and note_count > 0:
        if click.confirm(f"Treat existing notes as raw source material? ({note_count} files)"):
            console.print("[dim]Existing notes will be ingested as raw material.[/dim]")
            console.print("[dim]Run [bold]olw ingest --all[/bold] to process them.[/dim]")

    _write_vault_schema(vault)
    _write_index(vault)
    _cleanup_legacy_index(vault)


def _cleanup_legacy_index(vault: Path) -> None:
    """Remove wiki/INDEX.md if it's the bootstrap stub and is distinct from wiki/index.md."""
    old = vault / "wiki" / "INDEX.md"
    new = vault / "wiki" / "index.md"
    if not old.exists():
        return
    # On case-insensitive FS old and new are the same file — don't delete
    if new.exists():
        try:
            if old.samefile(new):
                return
        except OSError:
            return
    try:
        content = old.read_text(encoding="utf-8")
        if content == _INDEX_STUB:
            old.unlink()
    except Exception:
        pass


def _write_vault_schema(vault: Path) -> None:
    schema_path = vault / "vault-schema.md"
    if not schema_path.exists():
        schema_path.write_text(
            "# Vault Schema\n\n"
            "## Folder Structure\n"
            "- `raw/` — input notes (immutable, never edited by olw)\n"
            "- `wiki/` — AI-synthesised articles (managed by olw)\n"
            "- `wiki/.drafts/` — pending human review\n\n"
            "## Note Format\n"
            "Every wiki note has YAML frontmatter with: title, tags, sources, "
            "confidence, status, created, updated.\n\n"
            "## Links\n"
            "Use `[[Article Title]]` wikilinks between notes.\n"
        )


_INDEX_STUB = (
    "---\ntitle: Index\ntags: [index]\nstatus: published\n---\n\n"
    "# Wiki Index\n\n_Updated automatically by olw._\n"
)


def _write_index(vault: Path) -> None:
    index = vault / "wiki" / "index.md"
    if not index.exists():
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(_INDEX_STUB)


# ── setup ─────────────────────────────────────────────────────────────────────


def _pick_model(
    console: Console,
    client,
    step_label: str,
    description: str,
    default_fallback: str,
    connected: bool,
) -> str:
    """Interactive model selector — shows table if models available, else free-text."""
    console.print()
    console.print(f"  [bold]{step_label}[/bold]  {description}")

    models: list[dict] = []
    if connected:
        models = client.list_models_detailed()

    if models:
        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("#", style="dim", width=3)
        table.add_column("Model")
        table.add_column("Size", style="dim")
        for i, m in enumerate(models, 1):
            table.add_row(str(i), m["name"], m["size_gb"])
        console.print(table)
        console.print()
        raw = Prompt.ask("    Select (number or name)", default="1", console=console).strip()
        if not raw:
            return default_fallback
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(models):
                return models[idx]["name"]
            console.print(f"    [yellow]Invalid number, using {default_fallback}[/yellow]")
            return default_fallback
        return raw
    else:
        if connected:
            console.print(
                "    [yellow]No models found.[/yellow] "
                "Pull one first: [bold]ollama pull qwen3:4b[/bold]"
            )
        console.print("    (e.g. qwen3:4b, gemma3:4b, qwen3.6:35b-a3b)")
        raw = Prompt.ask("    Model name", default=default_fallback, console=console).strip()
        return raw if raw else default_fallback


@cli.command()
@click.option("--non-interactive", is_flag=True, help="Print current config and exit")
@click.option("--reset", is_flag=True, help="Clear saved config and re-run wizard")
@click.option(
    "--provider",
    "provider_preset",
    default=None,
    help="Skip provider selection (e.g. groq, lm_studio)",
)
def setup(non_interactive: bool, reset: bool, provider_preset: str | None):
    """Interactive wizard: configure provider, models, and default vault."""
    from .global_config import GlobalConfig, load_global_config, save_global_config
    from .providers import PROVIDER_REGISTRY, get_provider, list_all_providers

    # ── non-interactive: show current config ──────────────────────────────────
    if non_interactive:
        gcfg = load_global_config()
        if not gcfg:
            console.print(
                "[dim]No global config found. Run [bold]olw setup[/bold] to configure.[/dim]"
            )
            return
        table = Table(title="Global config", show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Value")
        prov_display = gcfg.provider_name or (gcfg.ollama_url and "ollama") or "[dim]not set[/dim]"
        table.add_row("Provider", prov_display)
        table.add_row("URL", gcfg.provider_url or gcfg.ollama_url or "[dim]not set[/dim]")
        table.add_row("API key", "***" if gcfg.api_key else "[dim]not set[/dim]")
        table.add_row("Fast model", gcfg.fast_model or "[dim]not set[/dim]")
        table.add_row("Heavy model", gcfg.heavy_model or "[dim]not set[/dim]")
        table.add_row("Default vault", gcfg.vault or "[dim]not set[/dim]")
        console.print(table)
        return

    # ── reset: wipe config before wizard ─────────────────────────────────────
    if reset:
        save_global_config(GlobalConfig())
        console.print("[dim]Config cleared.[/dim]")

    try:
        # ── Header ───────────────────────────────────────────────────────────
        console.print()
        from importlib.metadata import version as _pkg_version

        try:
            _ver = _pkg_version("obsidian-llm-wiki")
        except Exception:
            _ver = "unknown"
        console.print(
            Panel(
                f"[bold]obsidian-llm-wiki[/bold] v{_ver}  ·  setup",
                expand=False,
                border_style="blue",
                padding=(0, 4),
            )
        )
        console.print()

        all_providers = list_all_providers()

        # ── Step 1 — Provider selection ───────────────────────────────────────
        if provider_preset:
            chosen_prov = get_provider(provider_preset)
            if chosen_prov is None:
                console.print(
                    f"    [yellow]Unknown provider '{provider_preset}', using Ollama.[/yellow]"
                )
                chosen_prov = PROVIDER_REGISTRY["ollama"]
            chosen_name = chosen_prov.name
        else:
            console.print("  [bold]Step 1[/bold]  Provider\n")

            # Build numbered list
            local_provs = [p for p in all_providers if p.is_local]
            cloud_provs = [p for p in all_providers if not p.is_local and p.name != "custom"]

            console.print("    [bold]Local[/bold] (no API key needed):")
            idx_map: dict[int, str] = {}
            counter = 1
            for p in local_provs:
                marker = "  [default]" if p.name == "ollama" else ""
                console.print(f"      {counter:2}. {p.display_name:<14} {p.default_url}{marker}")
                idx_map[counter] = p.name
                counter += 1

            console.print()
            console.print("    [bold]Cloud[/bold] (API key required):")
            for p in cloud_provs:
                url_hint = p.default_url if p.default_url else "(enter URL manually)"
                console.print(f"      {counter:2}. {p.display_name:<14} {url_hint}")
                idx_map[counter] = p.name
                counter += 1

            console.print()
            console.print(f"      {counter:2}. Custom         (enter URL manually)")
            idx_map[counter] = "custom"

            console.print()
            raw = Prompt.ask(
                "    Select provider (number or name)", default="1", console=console
            ).strip()

            if raw.isdigit():
                num = int(raw)
                chosen_name = idx_map.get(num, "ollama")
            elif raw in PROVIDER_REGISTRY:
                chosen_name = raw
            else:
                console.print(f"    [yellow]Unknown '{raw}', defaulting to Ollama.[/yellow]")
                chosen_name = "ollama"

            chosen_prov = PROVIDER_REGISTRY[chosen_name]

        # ── Step 2 — URL ──────────────────────────────────────────────────────
        console.print()
        console.print("  [bold]Step 2[/bold]  URL")
        default_url = chosen_prov.default_url or ""
        if chosen_name == "azure":
            console.print(
                "    Azure format: https://{resource}.openai.azure.com/openai/deployments/{model}"
            )
        provider_url = Prompt.ask("    Base URL", default=default_url, console=console).strip()
        if not provider_url:
            provider_url = default_url
        if not provider_url and chosen_name in ("custom", "azure"):
            console.print(
                "    [red]URL is required for this provider. "
                "Run [bold]olw setup[/bold] again and enter a valid URL.[/red]"
            )
            sys.exit(1)

        # ── Step 3 — API key (cloud + custom providers) ───────────────────────
        import os

        needs_key_prompt = chosen_prov.requires_auth or chosen_name == "custom"
        api_key: str | None = None
        if needs_key_prompt:
            console.print()
            console.print("  [bold]Step 3[/bold]  API key")
            if chosen_prov.env_var:
                env_hint = f"  [dim](or set {chosen_prov.env_var} env var)[/dim]"
            elif chosen_name == "custom":
                env_hint = "  [dim](optional — press Enter to skip)[/dim]"
            else:
                env_hint = ""
            console.print(f"    API key{env_hint}")
            raw_key = Prompt.ask("    Key", default="", password=True, console=console).strip()
            api_key = raw_key if raw_key else None

        # ── Build a temp client to probe for model list ───────────────────────
        if chosen_name == "ollama":
            from .ollama_client import OllamaClient

            temp_client = OllamaClient(base_url=provider_url, timeout=5)
        else:
            from .openai_compat_client import OpenAICompatClient

            resolved_key = api_key
            if not resolved_key and chosen_prov.env_var:
                resolved_key = os.environ.get(chosen_prov.env_var)
            if not resolved_key:
                resolved_key = os.environ.get("OLW_API_KEY")
            temp_client = OpenAICompatClient(
                base_url=provider_url,
                provider_name=chosen_name,
                api_key=resolved_key,
                timeout=5,
                supports_json_mode=chosen_prov.supports_json_mode,
                supports_embeddings=chosen_prov.supports_embeddings,
                azure=chosen_prov.azure,
            )
        connected = temp_client.healthcheck()
        if connected:
            console.print("    [green]✓ connected[/green]")
        else:
            console.print(
                f"    [yellow]Warning:[/yellow] Cannot reach {provider_url} — continuing anyway."
            )

        # ── Default model names per provider ──────────────────────────────────
        # For non-Ollama providers, leave defaults empty — model names are
        # provider-specific and must be entered by the user.
        default_fast = "qwen3:4b" if chosen_name == "ollama" else ""
        default_heavy = "qwen3.6:35b-a3b" if chosen_name == "ollama" else ""
        if chosen_name != "ollama" and not connected:
            console.print(
                "    [dim]Tip: enter the model name exactly as the provider lists it "
                "(e.g. llama-3.1-70b-versatile for Groq).[/dim]"
            )

        step_offset = 1 if needs_key_prompt else 0

        # ── Step 4 — Fast model ───────────────────────────────────────────────
        fast_model = _pick_model(
            console=console,
            client=temp_client,
            step_label=f"Step {3 + step_offset}",
            description="Fast model  [dim](analysis & routing · 3–8B recommended)[/dim]",
            default_fallback=default_fast,
            connected=connected,
        )

        # ── Step 5 — Heavy model ──────────────────────────────────────────────
        heavy_model = _pick_model(
            console=console,
            client=temp_client,
            step_label=f"Step {4 + step_offset}",
            description="Heavy model  [dim](article writing · 7–14B recommended)[/dim]",
            default_fallback=default_heavy,
            connected=connected,
        )

        temp_client.close()

        # ── Final step — Default vault ────────────────────────────────────────
        console.print()
        step_label = f"Step {5 + step_offset}"
        console.print(
            f"  [bold]{step_label}[/bold]  Default vault path  [dim](press Enter to skip)[/dim]"
        )
        vault_input = Prompt.ask("    Vault path", default="", console=console)
        vault_path: str | None = None
        if vault_input.strip():
            vault_path = str(Path(vault_input).expanduser().resolve())

        # ── Save ──────────────────────────────────────────────────────────────
        # Preserve existing azure_api_version so re-running setup doesn't reset it.
        existing_cfg = load_global_config()
        if chosen_name == "azure":
            azure_api_ver = (
                existing_cfg.azure_api_version
                if existing_cfg and existing_cfg.azure_api_version
                else "2024-02-15-preview"
            )
        else:
            azure_api_ver = None

        # Keep ollama_url for backward compat when Ollama is selected
        cfg = GlobalConfig(
            vault=vault_path,
            ollama_url=provider_url if chosen_name == "ollama" else None,
            fast_model=fast_model if fast_model else None,
            heavy_model=heavy_model if heavy_model else None,
            provider_name=chosen_name,
            provider_url=provider_url,
            api_key=api_key,
            azure_api_version=azure_api_ver,
        )
        save_global_config(cfg)

        # ── Summary panel ─────────────────────────────────────────────────────
        init_target = vault_path or "~/my-wiki"
        summary_lines = [
            "[green]✓[/green]  Setup complete\n",
            f"  Provider:     [bold]{chosen_prov.display_name}[/bold]",
            f"  URL:          {provider_url}",
        ]
        if api_key:
            summary_lines.append("  API key:      ***")
        if fast_model:
            summary_lines.append(f"  Fast model:   [bold]{fast_model}[/bold]")
        if heavy_model:
            summary_lines.append(f"  Heavy model:  [bold]{heavy_model}[/bold]")
        if vault_path:
            summary_lines.append(f"  Vault:        {vault_path}")
        summary_lines += [
            "",
            "  Next steps:",
            f"    [bold]olw init {init_target}[/bold]",
            "    [bold]olw run[/bold]  (or: olw ingest --all && olw compile)",
        ]
        console.print()
        console.print(
            Panel("\n".join(summary_lines), border_style="green", expand=False, padding=(0, 2))
        )

    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Setup interrupted.[/yellow]")
        sys.exit(1)


# ── ingest ────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--all", "ingest_all", is_flag=True, help="Ingest all files in raw/")
@click.option("--force", is_flag=True, help="Re-ingest already-processed notes")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
def ingest(vault_str, ingest_all, force, paths):
    """Analyze raw notes: extract concepts, quality, suggested topics."""

    config = _load_config(vault_str)
    client, db = _load_deps(config)
    from .analytics import AnalyticsCollector, set_active_collector
    from .pipeline.ingest import collect_ingest_paths as _collect_ingest_paths

    if ingest_all:
        target_paths = _collect_ingest_paths(config)
    elif paths:
        target_paths = _collect_ingest_paths(config, [Path(p).resolve() for p in paths])
    else:
        click.echo("Specify --all or provide file paths.", err=True)
        sys.exit(1)

    if not target_paths:
        console.print("[yellow]No notes found in raw/[/yellow]")
        return

    _ingest_collector = AnalyticsCollector(db, config, "ingest")
    set_active_collector(_ingest_collector)

    durations: list[float] = []
    processed_durations: list[float] = []
    total_paths = len(target_paths)

    # Pre-scan: classify files by hash before showing the progress bar so that
    # the bar opens at the correct position and the ETA only counts new work.
    from .pipeline.ingest import _content_hash, ingest_note as _ingest_note
    from .vault import parse_note as _parse_note

    confirmed_done: list[Path] = []
    to_process: list[Path] = []

    if not force:
        ingested_recs = {rec.path: rec.content_hash for rec in db.list_raw(status="ingested")}
        console.print(f"[dim]Scanning {total_paths} files…[/dim]", end="\r")
        for path in target_paths:
            rel = path.relative_to(config.vault).as_posix()
            if rel in ingested_recs:
                try:
                    meta, body = _parse_note(path)
                    source_pdf = meta.get("source_pdf", "")
                    hash_input = (source_pdf + "\x00" + body) if source_pdf else body
                    if _content_hash(hash_input) == ingested_recs[rel]:
                        confirmed_done.append(path)
                        continue
                except Exception:
                    pass
            to_process.append(path)
    else:
        to_process = list(target_paths)

    pre_done_count = len(confirmed_done)
    new_count = len(to_process)
    skipped = pre_done_count
    ingested = failed = 0

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Ingesting...", total=total_paths, completed=pre_done_count)

            for new_idx, path in enumerate(to_process, 1):
                step_t0 = time.monotonic()
                overall_pct = (pre_done_count + new_idx - 1) / total_paths * 100
                progress.update(
                    task,
                    description=f"[dim]{path.name} | {overall_pct:5.1f}%"
                    f" | ETA {_format_eta(None)}[/dim]",
                )
                result = _ingest_note(
                    path=path,
                    config=config,
                    client=client,
                    db=db,
                    force=force,
                )
                if result is None:
                    # Distinguish skip vs failure by checking DB status
                    rel = str(path.relative_to(config.vault))
                    rec = db.get_raw(rel)
                    if rec and rec.status == "failed":
                        failed += 1
                    else:
                        skipped += 1
                else:
                    ingested += 1
                elapsed = time.monotonic() - step_t0
                durations.append(elapsed)
                if result is not None:
                    processed_durations.append(elapsed)
                # ETA counts down against new files only, not pre-done skips
                eta = None
                if new_idx < new_count and durations:
                    basis = processed_durations or durations
                    eta = (sum(basis) / len(basis)) * (new_count - new_idx)
                overall_pct = (pre_done_count + new_idx) / total_paths * 100
                progress.update(
                    task,
                    description=f"[dim]{path.name} | {overall_pct:5.1f}%"
                    f" | ETA {_format_eta(eta)}[/dim]",
                )
                progress.advance(task)
    except KeyboardInterrupt:
        console.print("\n[yellow]Ingest interrupted.[/yellow]")
        sys.exit(130)
    finally:
        set_active_collector(None)
        try:
            _jsonl = config.vault / ".olw" / "analytics.jsonl"
            _ingest_collector.flush(jsonl_path=_jsonl)
        except Exception:
            pass

    console.print(
        f"[green]Done.[/green] Ingested: {ingested}  Skipped: {skipped}  Failed: {failed}"
    )

    # Update index and log
    from .indexer import append_log, generate_index

    generate_index(config, db)
    if ingested:
        append_log(config, f"ingest | {ingested} notes ingested")

    if ingested and config.pipeline.auto_commit:
        from .git_ops import git_commit

        git_commit(
            config.vault,
            f"ingest: {ingested} notes",
            paths=["raw/", "wiki/sources/", "wiki/index.md", "wiki/log.md", "vault-schema.md"],
        )
        console.print("[dim]Git commit created.[/dim]")


# ── compile ───────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--dry-run", is_flag=True, help="Show plan, write nothing")
@click.option("--auto-approve", is_flag=True, help="Publish immediately (skip draft review)")
@click.option("--force", is_flag=True, help="Recompile even manually-edited articles")
@click.option("--legacy", is_flag=True, help="Use legacy LLM-planning compile (CompilePlan)")
@click.option(
    "--retry-failed",
    "retry_failed",
    is_flag=True,
    help="Re-ingest raw notes that previously failed, then compile",
)
def compile(vault_str, dry_run, auto_approve, force, legacy, retry_failed):
    """Synthesize ingested notes into wiki article drafts."""
    from .analytics import AnalyticsCollector, set_active_collector
    from .git_ops import git_commit
    from .pipeline.compile import approve_drafts, compile_concepts, compile_notes

    config = _load_config(vault_str)
    client, db = _load_deps(config)
    _compile_collector = AnalyticsCollector(db, config, "compile")
    set_active_collector(_compile_collector)

    # Re-ingest previously failed notes before compiling
    if retry_failed:
        failed_recs = db.list_raw(status="failed")
        if not failed_recs:
            console.print("[dim]No failed notes to retry.[/dim]")
        else:
            console.print(f"[yellow]Retrying {len(failed_recs)} failed note(s)...[/yellow]")
            from .pipeline.ingest import ingest_note as _ingest_note

            retried = 0
            for rec in failed_recs:
                p = config.vault / rec.path
                if not p.exists():
                    console.print(f"  [red]Not found, skipping:[/red] {rec.path}")
                    continue
                db.mark_raw_status(rec.path, "new")
                result = _ingest_note(path=p, config=config, client=client, db=db, force=True)
                if result is not None:
                    retried += 1
            console.print(f"[green]Re-ingested {retried}/{len(failed_recs)} note(s).[/green]")

    if dry_run:
        console.print("[dim]Dry run — no files will be written.[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        if legacy:
            task = progress.add_task("Planning compilation (legacy)...", total=None)
            draft_paths, failed = compile_notes(
                config=config,
                client=client,
                db=db,
                dry_run=dry_run,
            )
        else:
            task = progress.add_task("Compiling concepts...", total=1)
            compile_started = time.monotonic()
            compile_state = {"total": 1}

            def _on_progress(idx: int, total: int, name: str) -> None:
                completed = max(idx - 1, 0)
                compile_state["total"] = max(total, 1)
                eta = None
                if completed > 0 and total > completed:
                    elapsed = time.monotonic() - compile_started
                    eta = (elapsed / completed) * (total - completed)
                progress.update(
                    task,
                    total=total,
                    completed=completed,
                    description=f"[dim]{name} | {(completed / total) * 100:5.1f}%"
                    f" | ETA {_format_eta(eta)}[/dim]",
                )

            draft_paths, failed, _ = compile_concepts(
                config=config,
                client=client,
                db=db,
                force=force,
                dry_run=dry_run,
                on_progress=_on_progress,
            )
            final_total = compile_state["total"]
            progress.update(
                task,
                total=final_total,
                completed=final_total,
                description=f"[dim]Done | 100.0% | ETA {_format_eta(0)}[/dim]",
            )

    set_active_collector(None)
    try:
        _jsonl = config.vault / ".olw" / "analytics.jsonl"
        _compile_collector.flush(jsonl_path=None if dry_run else _jsonl)
    except Exception:
        pass

    if dry_run:
        return

    if draft_paths:
        console.print(f"\n[green]{len(draft_paths)} draft(s) written:[/green]")
        for p in draft_paths:
            console.print(f"  {p.relative_to(config.vault)}")

    if failed:
        console.print(f"[yellow]{len(failed)} article(s) failed:[/yellow] {', '.join(failed)}")

    # Update index and log
    from .indexer import append_log, generate_index

    generate_index(config, db)
    if draft_paths:
        append_log(config, f"compile | {len(draft_paths)} drafts written")

    if auto_approve and draft_paths:
        published = approve_drafts(config, db, draft_paths)
        generate_index(config, db)
        append_log(config, f"approve | {len(published)} articles published")
        if config.pipeline.auto_commit:
            git_commit(
                config.vault, f"compile: {len(published)} articles", paths=["wiki/", ".olw/"]
            )
        console.print(f"[green]Published {len(published)} articles.[/green]")
    elif draft_paths:
        console.print("\nReview drafts in [bold]wiki/.drafts/[/bold], then run:")
        console.print("  [bold]olw approve --all[/bold]")


# ── approve ───────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--all", "approve_all", is_flag=True)
@click.argument("files", nargs=-1, type=click.Path())
def approve(vault_str, approve_all, files):
    """Publish draft(s) from wiki/.drafts/ to wiki/."""
    from .git_ops import git_commit
    from .pipeline.compile import approve_drafts

    config = _load_config(vault_str)
    db = _load_db(config)

    if approve_all:
        paths = None  # approve_drafts handles all
    elif files:
        paths = [Path(f) for f in files]
    else:
        click.echo("Specify --all or file paths.", err=True)
        sys.exit(1)

    published = approve_drafts(config, db, paths)
    if not published:
        console.print("[yellow]No drafts to approve.[/yellow]")
        return

    console.print(f"[green]Published {len(published)} article(s).[/green]")

    # Update index and log
    from .indexer import append_log, generate_index

    generate_index(config, db)
    append_log(config, f"approve | {len(published)} articles published")

    if config.pipeline.auto_commit:
        git_commit(
            config.vault, f"approve: {len(published)} articles published", paths=["wiki/", ".olw/"]
        )
        console.print("[dim]Git commit created.[/dim]")


# ── reject ────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--all", "reject_all", is_flag=True, help="Reject all drafts in wiki/.drafts/")
@click.option("--feedback", default="", help="Reason for rejection")
@click.argument("files", nargs=-1, type=click.Path())
def reject(vault_str, reject_all, feedback, files):
    """Discard draft article(s) and store rejection feedback for future recompiles."""
    from .pipeline.compile import reject_draft

    config = _load_config(vault_str)
    db = _load_db(config)

    if reject_all:
        draft_paths = list(config.drafts_dir.rglob("*.md")) if config.drafts_dir.exists() else []
        if not draft_paths:
            console.print("[yellow]No drafts to reject.[/yellow]")
            return
        if not feedback:
            feedback = click.prompt("Reason for rejecting all drafts?", default="")
    elif files:
        draft_paths = [Path(f).resolve() for f in files]
        for p in draft_paths:
            if not p.exists():
                click.echo(f"File not found: {p}", err=True)
                sys.exit(1)
        if not feedback:
            feedback = click.prompt("Reason for rejection?", default="")
    else:
        click.echo("Specify --all or provide file paths.", err=True)
        sys.exit(1)

    from .vault import parse_note as _parse

    for draft_path in draft_paths:
        title = draft_path.stem
        try:
            meta, _ = _parse(draft_path)
            title = meta.get("title", draft_path.stem)
        except Exception:
            pass

        reject_draft(draft_path, config, db, feedback=feedback)
        console.print(f"[yellow]Draft rejected:[/yellow] {draft_path.name}")

        if feedback:
            count = db.rejection_count(title)
            if db.is_concept_blocked(title):
                console.print(
                    f"[red]⚠ '{title}' blocked after {count} rejections. "
                    f'Use [bold]olw unblock "{title}"[/bold] to re-enable.[/red]'
                )
            else:
                console.print(
                    f"[dim]Feedback saved. Next compile of '{title}' will address it. "
                    f"({count}/{db._REJECTION_CAP} rejections)[/dim]"
                )

    if len(draft_paths) > 1:
        console.print(f"[green]Rejected {len(draft_paths)} draft(s).[/green]")


# ── status ────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--failed", "show_failed", is_flag=True, help="List failed notes with error messages")
def status(vault_str, show_failed):
    """Show vault health, pending drafts, and pipeline stats."""
    config = _load_config(vault_str)
    db = _load_db(config)

    stats = db.stats()
    raw = stats.get("raw", {})

    table = Table(title="Vault Status", show_header=True)
    table.add_column("Category")
    table.add_column("Count", justify="right")

    table.add_row("Raw: new", str(raw.get("new", 0)))
    table.add_row("Raw: ingested", str(raw.get("ingested", 0)))
    table.add_row("Raw: compiled", str(raw.get("compiled", 0)))
    table.add_row("Raw: failed", str(raw.get("failed", 0)))
    table.add_row("Drafts pending", str(stats["drafts"]))
    table.add_row("Published articles", str(stats["published"]))

    console.print(table)

    # List pending drafts
    drafts = db.list_articles(drafts_only=True)
    if drafts:
        console.print(f"\n[bold]{len(drafts)} draft(s) pending review:[/bold]")
        for article in drafts:
            sources_str = ", ".join(Path(s).name for s in article.sources)
            console.print(f"  [dim]{article.path}[/dim]  (from: {sources_str})")
        console.print("\nRun [bold]olw approve --all[/bold] to publish.")

    # List failed notes if requested (or if there are any)
    if show_failed or raw.get("failed", 0):
        failed_recs = db.list_raw(status="failed")
        if failed_recs:
            console.print(f"\n[red][bold]{len(failed_recs)} failed note(s):[/bold][/red]")
            for rec in failed_recs:
                err = rec.error or "unknown error"
                console.print(f"  [dim]{rec.path}[/dim]")
                console.print(f"    [red]{err}[/red]")
            console.print("\nRun [bold]olw compile --retry-failed[/bold] to re-attempt.")

    # Show blocked concepts
    blocked = db.list_blocked_concepts()
    if blocked:
        console.print(f"\n[red][bold]{len(blocked)} blocked concept(s):[/bold][/red]")
        for concept in blocked:
            count = db.rejection_count(concept)
            console.print(f"  {concept} [dim]({count} rejections)[/dim]")
        console.print('[dim]Run [bold]olw unblock "Concept"[/bold] to re-enable.[/dim]')

    # Show pipeline lock status
    from .pipeline.lock import lock_holder_pid

    pid = lock_holder_pid(config.vault)
    if pid is not None:
        import os

        try:
            os.kill(pid, 0)
            console.print(f"\n[yellow]⚠ Pipeline lock held by PID {pid}[/yellow]")
        except (ProcessLookupError, PermissionError):
            console.print(f"\n[dim]Lock file present (PID {pid}) but process not running[/dim]")


# ── undo ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--steps", default=1, show_default=True)
def undo(vault_str, steps):
    """Revert last N [olw] auto-commits (uses git revert — safe)."""
    from .git_ops import git_undo

    config = _load_config(vault_str)
    reverted = git_undo(config.vault, steps=steps)
    if reverted:
        console.print(f"[green]Reverted {len(reverted)} commit(s):[/green]")
        for msg in reverted:
            console.print(f"  {msg}")
    else:
        console.print("[yellow]No [olw] commits found to revert.[/yellow]")


# ── clean ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
def clean(vault_str, yes):
    """Clear state DB, wiki/, and drafts — raw/ notes are kept.

    Use this to start fresh without deleting your source material.
    """
    import shutil

    config = _load_config(vault_str)

    targets = [
        ("state DB", config.state_db_path),
        ("wiki/", config.wiki_dir),
    ]

    console.print("[bold yellow]This will delete:[/bold yellow]")
    for label, path in targets:
        if path.exists():
            console.print(f"  {label}: {path}")
    console.print("Raw notes in [bold]raw/[/bold] are NOT touched.")

    if not yes:
        click.confirm("Proceed?", abort=True)

    for label, path in targets:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            console.print(f"  [dim]Deleted {label}[/dim]")

    # Re-create empty wiki/ structure
    config.wiki_dir.mkdir(parents=True, exist_ok=True)
    config.drafts_dir.mkdir(parents=True, exist_ok=True)
    config.sources_dir.mkdir(parents=True, exist_ok=True)

    console.print("[green]Clean complete.[/green] Run [bold]olw ingest --all[/bold] to restart.")


# ── doctor ───────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
def doctor(vault_str):
    """Check LLM provider connection, model availability, and vault health."""
    from .client_factory import LLMError, build_client

    config = _load_config(vault_str)
    db = _load_db(config)
    ok = True
    prov = config.effective_provider

    console.print("[bold]olw doctor[/bold]\n")

    # ── Vault structure ──────────────────────────────────────────────────────
    console.print("[bold]Vault structure[/bold]")
    toml_path = config.vault / "wiki.toml"
    if not toml_path.exists():
        console.print(
            f"  [red]✗[/red] wiki.toml missing — vault not initialised.\n"
            f"    Run: [bold]olw init {config.vault}[/bold]"
        )
        console.print("\n[red][bold]Vault not initialised. Remaining checks skipped.[/bold][/red]")
        sys.exit(1)

    checks = {
        "raw/": config.raw_dir,
        "wiki/": config.wiki_dir,
        "wiki/.drafts/": config.drafts_dir,
        "wiki/sources/": config.sources_dir,
        ".olw/": config.olw_dir,
        "wiki.toml": toml_path,
    }
    for name, path in checks.items():
        if path.exists():
            console.print(f"  [green]✓[/green] {name}")
        else:
            console.print(f"  [yellow]![/yellow] {name} missing (run [bold]olw init[/bold])")

    # ── Provider connection ───────────────────────────────────────────────────
    console.print(f"\n[bold]{prov.name}[/bold]")
    client = build_client(config)
    try:
        client.require_healthy()
        console.print(f"  [green]✓[/green] Reachable at {prov.url}")
    except LLMError as e:
        console.print(f"  [red]✗[/red] {e}")
        ok = False

    # ── Model availability ────────────────────────────────────────────────────
    console.print("\n[bold]Models[/bold]")
    try:
        available_models = client.list_models()
    except Exception:
        available_models = []

    for label, model_name in [("fast", config.models.fast), ("heavy", config.models.heavy)]:
        if any(model_name in a for a in available_models):
            console.print(f"  [green]✓[/green] {label}: {model_name}")
        else:
            pull_hint = (
                f"run: [bold]ollama pull {model_name}[/bold]"
                if prov.name == "ollama"
                else "check provider model list"
            )
            console.print(f"  [yellow]![/yellow] {label}: {model_name} not found — {pull_hint}")
            ok = False

    # ── Vault stats ───────────────────────────────────────────────────────────
    console.print("\n[bold]Vault stats[/bold]")
    stats = db.stats()
    raw = stats.get("raw", {})
    console.print(f"  Raw notes:         {sum(raw.values())}")
    console.print(f"  Ingested:          {raw.get('ingested', 0) + raw.get('compiled', 0)}")
    console.print(f"  Drafts pending:    {stats['drafts']}")
    console.print(f"  Published:         {stats['published']}")

    console.print()
    if ok:
        console.print("[green][bold]All checks passed.[/bold][/green]")
    else:
        console.print("[yellow][bold]Some checks need attention (see above).[/bold][/yellow]")


# ── config ───────────────────────────────────────────────────────────────────


@cli.group()
def config():
    """Inspect and validate olw configuration."""


@config.command("show")
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
def config_show(vault_str):
    """Show effective merged configuration (global + vault wiki.toml)."""
    from .config import Config
    from .global_config import GlobalConfig, _global_config_path, load_global_config

    # ── Global config ─────────────────────────────────────────────────────────
    gcfg_path = _global_config_path()
    console.print("[bold]Global config[/bold]", f"[dim]{gcfg_path}[/dim]")
    gcfg: GlobalConfig | None = load_global_config()
    if not gcfg_path.exists():
        console.print("  [dim](not found — defaults used)[/dim]")
    elif gcfg is None:
        console.print("  [red]✗ malformed — see warning above[/red]")
    else:
        g_table = Table(show_header=False, box=None, padding=(0, 2))
        for field, val in gcfg.model_dump(exclude_none=True).items():
            display = "****" if field == "api_key" else str(val)
            g_table.add_row(f"[dim]{field}[/dim]", display)
        if g_table.row_count:
            console.print(g_table)
        else:
            console.print("  [dim](empty — all defaults)[/dim]")

    # ── Vault config ──────────────────────────────────────────────────────────
    console.print()
    if vault_str is None and gcfg and gcfg.vault:
        vault_str = gcfg.vault
    if vault_str is None:
        # Try auto-detect
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            if (parent / "wiki.toml").exists():
                vault_str = str(parent)
                break

    if vault_str is None:
        console.print("[bold]Vault config[/bold]  [dim](no vault — use --vault or cd into one)[/dim]")
        return

    vault_path = Path(vault_str).expanduser().resolve()
    toml_path = vault_path / "wiki.toml"
    console.print("[bold]Vault config[/bold]", f"[dim]{toml_path}[/dim]")

    if not toml_path.exists():
        console.print("  [yellow]! wiki.toml not found — run olw init[/yellow]")
        return

    try:
        cfg = Config.from_vault(vault_path)
    except Exception as e:
        console.print(f"  [red]✗ failed to load: {e}[/red]")
        return

    prov = cfg.effective_provider
    v_table = Table(show_header=False, box=None, padding=(0, 2))
    v_table.add_row("[dim]models.fast[/dim]", cfg.models.fast)
    v_table.add_row("[dim]models.heavy[/dim]", cfg.models.heavy)
    v_table.add_row("[dim]provider.name[/dim]", prov.name)
    v_table.add_row("[dim]provider.url[/dim]", prov.url)
    v_table.add_row("[dim]provider.timeout[/dim]", f"{prov.timeout:.0f}s")
    v_table.add_row("[dim]provider.fast_ctx[/dim]", str(prov.fast_ctx))
    v_table.add_row("[dim]provider.heavy_ctx[/dim]", str(prov.heavy_ctx))
    v_table.add_row("[dim]pipeline.auto_approve[/dim]", str(cfg.pipeline.auto_approve))
    v_table.add_row("[dim]pipeline.auto_commit[/dim]", str(cfg.pipeline.auto_commit))
    v_table.add_row("[dim]pipeline.ingest_parallel[/dim]", str(cfg.pipeline.ingest_parallel))
    v_table.add_row("[dim]pipeline.language[/dim]", cfg.pipeline.language or "(auto-detect)")
    v_table.add_row("[dim]pipeline.telemetry_enabled[/dim]", str(cfg.pipeline.telemetry_enabled))
    console.print(v_table)


@config.command("validate")
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
def config_validate(vault_str):
    """Check wiki.toml for unknown keys, type errors, and missing values."""
    import tomllib

    from .config import (
        ModelsConfig,
        OllamaConfig,
        PipelineConfig,
        ProviderConfig,
        RagConfig,
    )
    from .global_config import GlobalConfig, _global_config_path, load_global_config

    issues: list[str] = []
    ok_msgs: list[str] = []

    # ── Global config ─────────────────────────────────────────────────────────
    gcfg_path = _global_config_path()
    console.print("[bold]Validating global config[/bold]", f"[dim]{gcfg_path}[/dim]")
    if not gcfg_path.exists():
        console.print("  [dim]Not found — OK (optional)[/dim]")
    else:
        try:
            with open(gcfg_path, "rb") as f:
                raw_gcfg = tomllib.load(f)
            known_global = set(GlobalConfig.model_fields.keys())
            for key in raw_gcfg:
                if key not in known_global:
                    issues.append(f"global config: unknown key [bold]{key!r}[/bold] (typo?)")
            GlobalConfig(**raw_gcfg)
            ok_msgs.append("global config syntax OK")
        except Exception as e:
            issues.append(f"global config: {e}")

    # ── Vault config ──────────────────────────────────────────────────────────
    if vault_str is None:
        gcfg = load_global_config()
        if gcfg and gcfg.vault:
            vault_str = gcfg.vault
    if vault_str is None:
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            if (parent / "wiki.toml").exists():
                vault_str = str(parent)
                break

    console.print()
    if vault_str is None:
        console.print("[dim]No vault to validate — use --vault or cd into one.[/dim]")
    else:
        vault_path = Path(vault_str).expanduser().resolve()
        toml_path = vault_path / "wiki.toml"
        console.print("[bold]Validating vault config[/bold]", f"[dim]{toml_path}[/dim]")

        if not toml_path.exists():
            issues.append("wiki.toml not found — run olw init")
        else:
            try:
                with open(toml_path, "rb") as f:
                    raw = tomllib.load(f)

                known_top = {"models", "ollama", "provider", "pipeline", "rag"}
                section_models = {
                    "models": set(ModelsConfig.model_fields.keys()),
                    "ollama": set(OllamaConfig.model_fields.keys()),
                    "provider": set(ProviderConfig.model_fields.keys()),
                    "pipeline": set(PipelineConfig.model_fields.keys()),
                    "rag": set(RagConfig.model_fields.keys()),
                }

                for key in raw:
                    if key not in known_top:
                        issues.append(f"wiki.toml: unknown top-level key [bold]{key!r}[/bold] (typo?)")

                for section, known_keys in section_models.items():
                    if section in raw and isinstance(raw[section], dict):
                        for k in raw[section]:
                            if k not in known_keys:
                                issues.append(
                                    f"wiki.toml [{section}]: unknown key [bold]{k!r}[/bold]"
                                    f" — did you mean one of: {', '.join(sorted(known_keys)[:5])}?"
                                )

                # Full load validation
                from .config import Config
                Config.from_vault(vault_path)
                ok_msgs.append("wiki.toml syntax and schema OK")

            except Exception as e:
                issues.append(f"wiki.toml: {e}")

    # ── Report ────────────────────────────────────────────────────────────────
    console.print()
    for msg in ok_msgs:
        console.print(f"  [green]✓[/green] {msg}")
    for issue in issues:
        console.print(f"  [red]✗[/red] {issue}")

    if not issues:
        console.print("\n[green][bold]All checks passed.[/bold][/green]")
    else:
        console.print(f"\n[red][bold]{len(issues)} issue(s) found.[/bold][/red]")
        sys.exit(1)


# ── query ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--save", is_flag=True, help="Save answer to wiki/queries/")
@click.argument("question")
def query(vault_str, question, save):
    """Answer a question using your wiki as context (no embeddings needed)."""
    from rich.markdown import Markdown

    from .pipeline.query import run_query

    config = _load_config(vault_str)
    client, db = _load_deps(config)

    with console.status("[bold]Searching wiki index…"):
        answer, pages = run_query(config, client, db, question, save=save)

    if pages:
        console.print(f"[dim]Sources: {', '.join(pages)}[/dim]")
    console.print()
    console.print(Markdown(answer))
    if save:
        console.print("\n[green]Answer saved to wiki/queries/[/green]")


# ── lint ──────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--fix", is_flag=True, help="Auto-fix simple issues (missing frontmatter fields)")
def lint(vault_str, fix):
    """Check wiki health: orphans, broken links, missing frontmatter, low confidence."""
    from .pipeline.lint import run_lint

    config = _load_config(vault_str)
    db = _load_db(config)

    result = run_lint(config, db, fix=fix)

    # Score bar
    score = result.health_score
    colour = "green" if score >= 80 else "yellow" if score >= 50 else "red"
    console.print(f"\n[bold {colour}]Health: {score}/100[/bold {colour}]  {result.summary}")

    if result.issues:
        console.print()
        _TYPE_ICON = {
            "orphan": "○",
            "broken_link": "⛓",
            "missing_frontmatter": "⚙",
            "stale": "✎",
            "low_confidence": "↓",
        }
        from rich.markup import escape

        for iss in result.issues:
            icon = _TYPE_ICON.get(iss.issue_type, "!")
            fix_tag = " [dim][auto-fixable][/dim]" if iss.auto_fixable else ""
            console.print(f"  {icon} [bold]{iss.issue_type}[/bold]{fix_tag}  {escape(iss.path)}")
            console.print(f"     {escape(iss.description)}")
            console.print(f"     [dim]→ {escape(iss.suggestion)}[/dim]")
        console.print()

    if fix:
        fixed = sum(1 for i in result.issues if i.auto_fixable)
        if fixed:
            console.print(f"[green]Auto-fixed {fixed} issue(s).[/green]")


# ── metrics ──────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--last", default=0, help="Limit to last N events (0 = all)")
def metrics(vault_str, last):
    """Summarise telemetry: LLM success rates, latency, slowest concepts."""
    import json

    from .telemetry import resolve_metrics_path

    config = _load_config(vault_str)
    metrics_path = resolve_metrics_path(config)

    if not metrics_path.exists():
        console.print(f"[yellow]No telemetry file found at {metrics_path}[/yellow]")
        console.print("[dim]Telemetry is written when pipeline runs with telemetry_enabled = true.[/dim]")
        return

    events: list[dict] = []
    with metrics_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if last:
        events = events[-last:]

    if not events:
        console.print("[dim]No events in telemetry file.[/dim]")
        return

    console.print(f"[bold]Telemetry summary[/bold]  [dim]{metrics_path}[/dim]")
    console.print(f"[dim]{len(events)} event(s){f', last {last}' if last else ''}[/dim]\n")

    # ── LLM request stats ─────────────────────────────────────────────────────
    llm_events = [e for e in events if e.get("event_type") == "llm_request"]
    if llm_events:
        total = len(llm_events)
        successes = sum(1 for e in llm_events if e.get("success"))
        failures = total - successes
        latencies = [e["elapsed_ms"] for e in llm_events if e.get("elapsed_ms") and e.get("success")]
        avg_ms = sum(latencies) / len(latencies) if latencies else 0
        p95_ms = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

        console.print("[bold]LLM requests[/bold]")
        req_table = Table(show_header=False, box=None, padding=(0, 2))
        req_table.add_row("total", str(total))
        req_table.add_row("success", f"[green]{successes}[/green]")
        req_table.add_row("failed", f"[{'red' if failures else 'dim'}]{failures}[/{'red' if failures else 'dim'}]")
        req_table.add_row(
            "success rate",
            f"{'[green]' if successes / total >= 0.9 else '[yellow]'}{successes / total:.0%}[/{'green' if successes / total >= 0.9 else 'yellow'}]",
        )
        if latencies:
            req_table.add_row("avg latency", f"{avg_ms:.0f} ms")
            req_table.add_row("p95 latency", f"{p95_ms:.0f} ms")
        console.print(req_table)

        # By stage
        stages: dict[str, dict[str, int]] = {}
        for e in llm_events:
            stage = e.get("stage", "unknown")
            stages.setdefault(stage, {"ok": 0, "fail": 0})
            if e.get("success"):
                stages[stage]["ok"] += 1
            else:
                stages[stage]["fail"] += 1

        if len(stages) > 1:
            console.print("\n[bold]By stage[/bold]")
            st_table = Table("stage", "ok", "fail", "rate", box=None, padding=(0, 2))
            for stage, counts in sorted(stages.items()):
                stotal = counts["ok"] + counts["fail"]
                rate = counts["ok"] / stotal
                color = "green" if rate >= 0.9 else "yellow" if rate >= 0.7 else "red"
                st_table.add_row(
                    stage,
                    str(counts["ok"]),
                    str(counts["fail"]),
                    f"[{color}]{rate:.0%}[/{color}]",
                )
            console.print(st_table)

    # ── Function timings ──────────────────────────────────────────────────────
    timing_events = [e for e in events if e.get("event_type") == "function_timing"]
    if timing_events:
        console.print("\n[bold]Pipeline timings[/bold]")
        fn_times: dict[str, list[float]] = {}
        for e in timing_events:
            fn = e.get("function_name", "unknown")
            dur = e.get("duration_s")
            if dur is not None:
                fn_times.setdefault(fn, []).append(float(dur))

        tm_table = Table("function", "runs", "avg (s)", "max (s)", box=None, padding=(0, 2))
        for fn, times in sorted(fn_times.items()):
            tm_table.add_row(
                fn,
                str(len(times)),
                f"{sum(times) / len(times):.1f}",
                f"{max(times):.1f}",
            )
        console.print(tm_table)

    # ── Recent failures ───────────────────────────────────────────────────────
    recent_failures = [e for e in llm_events[-50:] if not e.get("success")]
    if recent_failures:
        console.print(f"\n[bold]Recent failures[/bold]  [dim](last {len(recent_failures)} of 50 checked)[/dim]")
        fail_table = Table("stage", "model", "timestamp", box=None, padding=(0, 2))
        for e in recent_failures[-10:]:
            ts = e.get("timestamp", "")[:19].replace("T", " ")
            fail_table.add_row(
                e.get("stage", "?"),
                e.get("model", "?"),
                ts,
            )
        console.print(fail_table)


# ── watch ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option(
    "--auto-approve", is_flag=True, help="Publish drafts immediately without manual review"
)
def watch(vault_str, auto_approve):
    """Watch raw/ for new/changed notes → auto-ingest + compile."""
    from .pipeline.lock import pipeline_lock
    from .pipeline.orchestrator import PipelineOrchestrator
    from .watcher import watch as _watch

    config = _load_config(vault_str)
    client, db = _load_deps(config)
    orchestrator = PipelineOrchestrator(config, client, db)

    debounce = config.pipeline.watch_debounce
    console.print(f"[bold]Watching[/bold] {config.raw_dir}  (debounce={debounce:.0f}s)")
    console.print("[dim]Ctrl+C to stop.[/dim]\n")

    def _on_event(paths: list[str]) -> None:
        md_paths = [p for p in paths if p.endswith(".md")]
        if not md_paths:
            return

        console.rule(f"[dim]{len(md_paths)} file(s) changed[/dim]")

        with pipeline_lock(config.vault) as acquired:
            if not acquired:
                console.print("[yellow]⚠ compile skipped — pipeline already running[/yellow]")
                return
            try:
                report = orchestrator.run(
                    paths=md_paths,
                    auto_approve=auto_approve or config.pipeline.auto_approve,
                    fix=config.pipeline.auto_maintain,
                )
            except Exception as exc:
                console.print(f"[red]Pipeline error:[/red] {exc}")
                return

        if report.ingested:
            console.print(f"  [green]✓[/green] ingested {report.ingested} note(s)")
        if report.compiled:
            console.print(f"  [green]✓[/green] {report.compiled} draft(s) compiled")
        if report.failed:
            failed_str = ", ".join(report.failed_names)
            console.print(
                f"  [yellow]![/yellow] {len(report.failed)} concept(s) failed: {failed_str}"
            )
        if report.published:
            console.print(f"  [green]✓[/green] {report.published} article(s) published")
        elif report.compiled:
            console.print("  [dim]Run [bold]olw approve --all[/bold] to publish drafts.[/dim]")
        if report.stubs_created:
            console.print(f"  [dim]Created {report.stubs_created} stub(s) for broken links[/dim]")

    _watch(config=config, client=client, db=db, on_event=_on_event)


# ── run ───────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--auto-approve", is_flag=True, help="Publish drafts immediately")
@click.option("--skip-bundles", is_flag=True, help="Skip bundle generation stage")
@click.option("--fix", is_flag=True, help="Create stubs for broken wikilinks")
@click.option("--max-rounds", default=2, show_default=True, help="Max compile rounds")
@click.option("--dry-run", is_flag=True, help="Report what would happen, make no changes")
def run(vault_str, auto_approve, skip_bundles, fix, max_rounds, dry_run):
    """Run full pipeline: ingest → compile → lint → [approve] → [bundles]."""
    from .pipeline.lock import pipeline_lock
    from .pipeline.orchestrator import PipelineOrchestrator

    config = _load_config(vault_str)
    client, db = _load_deps(config)

    if dry_run:
        console.print("[dim]Dry run — no changes will be made.[/dim]\n")

    with pipeline_lock(config.vault) as acquired:
        if not acquired:
            err_console.print("Pipeline already running — lock held. Check `olw status`.")
            sys.exit(1)
        orchestrator = PipelineOrchestrator(config, client, db)
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Running pipeline...", total=1)

                def _on_progress(
                    stage: str,
                    completed: int,
                    total: int,
                    eta_seconds: float | None,
                    detail: str,
                ) -> None:
                    stage_label = {
                        "ingest": "Ingest",
                        "compile_r1": "Compile r1",
                        "compile_r2": "Compile r2",
                    }.get(stage, stage)
                    safe_total = total if total > 0 else 1
                    pct = (completed / total) * 100 if total > 0 else 100.0
                    progress.update(
                        task,
                        total=safe_total,
                        completed=min(completed, safe_total),
                        description=f"[dim]{stage_label}: {detail} | {pct:5.1f}%"
                        f" | ETA {_format_eta(eta_seconds)}[/dim]",
                    )

                report = orchestrator.run(
                    auto_approve=auto_approve,
                    build_bundles=not skip_bundles,
                    fix=fix,
                    max_rounds=max_rounds,
                    dry_run=dry_run,
                    on_progress=_on_progress,
                    analytics_jsonl_path=str(config.vault / ".olw" / "analytics.jsonl"),
                )
                progress.update(
                    task,
                    total=1,
                    completed=1,
                    description=f"[dim]Done | 100.0% | ETA {_format_eta(0)}[/dim]",
                )
        except KeyboardInterrupt:
            console.print("\n[yellow]Run interrupted.[/yellow]")
            sys.exit(130)

    table = Table(title="Pipeline Report", show_header=True)
    table.add_column("Step")
    table.add_column("Count", justify="right")
    table.add_column("Time", justify="right")

    table.add_row("Ingested", str(report.ingested), f"{report.timings.get('ingest', 0):.1f}s")
    table.add_row(
        "Compiled",
        str(report.compiled),
        f"{report.timings.get('compile_r1', 0) + report.timings.get('compile_r2', 0):.1f}s",
    )
    table.add_row("Published", str(report.published), "")
    table.add_row("Bundles", str(report.bundles_created), "")
    table.add_row("Lint issues", str(report.lint_issues), "")
    table.add_row("Stubs created", str(report.stubs_created), "")
    if report.rounds > 1:
        table.add_row("Compile rounds", str(report.rounds), "")
    console.print(table)

    if report.failed:
        console.print(f"\n[yellow]{len(report.failed)} concept(s) failed:[/yellow]")
        for f in report.failed:
            console.print(f"  [dim]{f.concept}[/dim] ({f.reason.value})")


# ── review ────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
def review(vault_str):
    """Interactive draft review: approve, reject, edit, or diff drafts."""

    from .pipeline.compile import approve_drafts, reject_draft
    from .pipeline.review import (
        compute_diff,
        compute_rejection_diff,
        list_drafts,
        load_draft_content,
    )

    config = _load_config(vault_str)
    db = _load_db(config)

    while True:
        summaries = list_drafts(config, db)
        if not summaries:
            console.print("[dim]No drafts pending review.[/dim]")
            return

        # Build menu table
        table = Table(title="Drafts Pending Review", show_header=True, show_lines=False)
        table.add_column("#", justify="right", style="dim")
        table.add_column("Title")
        table.add_column("Conf", justify="right")
        table.add_column("Sources", justify="right")
        table.add_column("Rejections", justify="right")
        table.add_column("Flags", justify="left")

        for i, s in enumerate(summaries, 1):
            flags = ""
            if s.has_annotations:
                flags += "⚠ annotations  "
            if s.rejection_count > 0:
                flags += f"{'🔴' if s.rejection_count >= 3 else '🟡'} rejected"
            conf_color = (
                "green" if s.confidence >= 0.6 else "yellow" if s.confidence >= 0.4 else "red"
            )
            table.add_row(
                str(i),
                s.title,
                f"[{conf_color}]{s.confidence:.2f}[/{conf_color}]",
                str(s.source_count),
                str(s.rejection_count),
                flags.strip(),
            )

        console.print(table)
        console.print("\n[dim]  Type: number=open draft, a=approve all, x=reject all, q=quit[/dim]")
        choice = click.prompt("\nChoice", prompt_suffix=" > ").strip().lower()

        if choice == "q":
            return
        elif choice == "a":
            all_paths = [s.path for s in summaries]
            published = approve_drafts(config, db, all_paths)
            console.print(f"[green]Published {len(published)} article(s).[/green]")
            from .indexer import append_log, generate_index

            generate_index(config, db)
            append_log(config, f"review | approved {len(published)} articles")
            return
        elif choice == "x":
            reason = click.prompt("Reason for rejecting all", default="")
            for s in summaries:
                reject_draft(s.path, config, db, feedback=reason)
            console.print(f"[yellow]Rejected {len(summaries)} draft(s).[/yellow]")
            return
        elif choice.isdigit():
            idx = int(choice) - 1
            if idx < 0 or idx >= len(summaries):
                console.print("[red]Invalid selection.[/red]")
                continue
            _review_single(
                summaries[idx],
                config,
                db,
                approve_drafts,
                reject_draft,
                compute_diff,
                compute_rejection_diff,
                load_draft_content,
            )
        else:
            console.print("[red]Unknown command.[/red]")


def _review_single(
    summary,
    config,
    db,
    approve_drafts,
    reject_draft,
    compute_diff,
    compute_rejection_diff,
    load_draft_content,
):
    """Handle single-draft review loop."""
    from rich.panel import Panel

    from .vault import sanitize_filename

    while True:
        if not summary.path.exists():
            console.print("[yellow]Draft no longer exists.[/yellow]")
            return

        try:
            meta, body = load_draft_content(summary.path)
        except Exception as e:
            console.print(f"[red]Could not read draft: {e}[/red]")
            return

        # Show rejection history
        rejections = db.get_rejections(summary.title, limit=3)
        if rejections:
            console.print(
                Panel(
                    "\n".join(f"• {r['feedback']}" for r in rejections),
                    title=f"[red]Previous rejections ({len(rejections)})[/red]",
                    border_style="red",
                )
            )

        # Show metadata
        console.print(
            f"[bold]{summary.title}[/bold]  "
            f"conf={meta.get('confidence', 0):.2f}  "
            f"sources={summary.source_count}  "
            f"rejections={summary.rejection_count}"
        )

        # Show body
        console.print(Panel(body[:3000] + ("…" if len(body) > 3000 else ""), title="Draft"))

        console.print(
            "\n[dim]Type: a=approve, r=reject, e=edit, "
            "d=diff vs published, v=rejection diff, s=skip[/dim]"
        )
        raw_action = click.prompt("\nAction", prompt_suffix=" > ").strip()
        action = raw_action.lower()

        if action == "s":
            return
        elif action == "a":
            if not summary.path.exists():
                console.print("[yellow]Draft disappeared.[/yellow]")
                return
            published = approve_drafts(config, db, [summary.path])
            console.print(f"[green]Published:[/green] {published[0].name if published else '?'}")
            from .indexer import append_log, generate_index

            generate_index(config, db)
            append_log(config, f"review | approved {summary.title}")
            return
        elif action == "r":
            reason = click.prompt("Reason?", default="")
            if not summary.path.exists():
                console.print("[yellow]Draft disappeared.[/yellow]")
                return
            reject_draft(summary.path, config, db, feedback=reason)
            console.print("[yellow]Rejected.[/yellow]")
            if reason:
                count = db.rejection_count(summary.title)
                if db.is_concept_blocked(summary.title):
                    console.print(f"[red]⚠ '{summary.title}' is now blocked.[/red]")
                else:
                    console.print(f"[dim]({count}/{db._REJECTION_CAP} rejections)[/dim]")
            return
        elif action == "e":
            import os
            import subprocess

            editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
            subprocess.call([editor, str(summary.path)])
        elif action == "d":
            safe_name = sanitize_filename(summary.title)
            wiki_path = config.wiki_dir / f"{safe_name}.md"
            diff = compute_diff(summary.path, wiki_path)
            if diff is None:
                console.print("[dim]No published version — this is a new article.[/dim]")
            else:
                console.print(diff)
        elif action == "v":
            diff = compute_rejection_diff(summary.path, db, summary.title)
            if diff is None:
                console.print("[dim]No rejected body stored for this concept.[/dim]")
            else:
                console.print(diff)
        else:
            console.print("[red]Unknown action.[/red]")


# ── maintain ──────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option(
    "--fix", is_flag=True, help="Auto-fix missing frontmatter, invalid tags, create stubs"
)  # noqa: E501
@click.option("--stubs-only", is_flag=True, help="Only create stub articles")
@click.option("--dry-run", is_flag=True, help="Report issues without making changes")
def maintain(vault_str, fix, stubs_only, dry_run):
    """Wiki maintenance: lint, stub creation, orphan suggestions, concept merge hints."""
    from .pipeline.lint import run_lint
    from .pipeline.lock import pipeline_lock
    from .pipeline.maintain import (
        create_stubs,
        fix_broken_links,
        normalize_published_alias_links,
        suggest_concept_merges,
        suggest_orphan_links,
    )

    config = _load_config(vault_str)
    db = _load_db(config)

    if dry_run:
        console.print("[dim]Dry run — no changes will be made.[/dim]\n")

    with pipeline_lock(config.vault) as acquired:
        if not acquired:
            err_console.print("Pipeline already running — lock held.")
            sys.exit(1)

        # Quality warning
        quality = db.quality_stats()
        total_sources = sum(quality.values())
        if total_sources > 0:
            low_pct = round(100 * quality["low"] / total_sources)
            if low_pct > 60:
                console.print(
                    f"[yellow]⚠ {low_pct}% of sources are low quality — "
                    f"articles will have low confidence.[/yellow]"
                )

        # Blocked concepts
        blocked = db.list_blocked_concepts()
        if blocked:
            console.print(f"\n[red]{len(blocked)} blocked concept(s):[/red]")
            for concept in blocked:
                count = db.rejection_count(concept)
                console.print(f"  {concept} ({count} rejections)")
            console.print('[dim]Use [bold]olw unblock "Concept"[/bold] to re-enable.[/dim]')

        if stubs_only:
            if not dry_run:
                created = create_stubs(config, db)
                console.print(f"[green]Created {len(created)} stub(s).[/green]")
            else:
                result = run_lint(config, db)
                broken = [i for i in result.issues if i.issue_type == "broken_link"]
                console.print(f"[dim]Would create up to {min(len(broken), 5)} stub(s).[/dim]")
            return

        # Full lint
        result = run_lint(config, db, fix=fix and not dry_run)
        score = result.health_score
        colour = "green" if score >= 80 else "yellow" if score >= 50 else "red"
        console.print(f"\n[bold {colour}]Health: {score}/100[/bold {colour}]  {result.summary}")

        if result.issues:
            console.print()
            for iss in result.issues:
                fix_tag = " [dim][auto-fixable][/dim]" if iss.auto_fixable else ""
                console.print(f"  [bold]{iss.issue_type}[/bold]{fix_tag}  {iss.path}")
                console.print(f"    {iss.description}")

        # Alias link normalization in published articles (fix [[Alias]] → [[Canonical|Alias]])
        # Runs independently of broken-link detection: lint resolves aliases so they never
        # appear as broken, but published articles may still have raw alias-form links.
        if fix and not stubs_only:
            alias_normalized = normalize_published_alias_links(config, db, dry_run=dry_run)
            if alias_normalized:
                console.print(
                    f"\n[green]Normalized alias links in {alias_normalized} article(s).[/green]"
                )

        # Broken link repair + stub creation
        broken = [i for i in result.issues if i.issue_type == "broken_link"]
        if broken:
            if fix and not stubs_only:
                repair = fix_broken_links(config, db, broken, dry_run=dry_run)
                if repair.repaired:
                    console.print(f"\n[green]Repaired {repair.repaired} broken link(s).[/green]")
                remaining = repair.still_broken
                if remaining and not dry_run:
                    created = create_stubs(config, db, broken_link_issues=remaining)
                    if created:
                        console.print(f"[green]Created {len(created)} stub(s).[/green]")
                elif remaining:
                    console.print(
                        f"[dim]{len(remaining)} link(s) unresolvable"
                        f" — stubs would be created.[/dim]"
                    )
            elif fix:
                created = create_stubs(config, db, broken_link_issues=broken)
                if created:
                    console.print(f"\n[green]Created {len(created)} stub(s).[/green]")
            else:
                console.print(
                    f"\n[dim]{len(broken)} broken link(s) — "
                    f"run [bold]olw maintain --fix[/bold] to repair or create stubs.[/dim]"
                )

        # Orphan suggestions
        orphan_suggestions = suggest_orphan_links(config, db)
        if orphan_suggestions:
            console.print(f"\n[bold]Orphan link suggestions ({len(orphan_suggestions)}):[/bold]")
            for title, mentioners in orphan_suggestions[:5]:
                console.print(f"  {title} — mentioned in:")
                for m in mentioners[:3]:
                    console.print(f"    [dim]{m}[/dim]")

        # Concept merge suggestions
        merges = suggest_concept_merges(config, db)
        if merges:
            console.print(f"\n[bold]Possible concept duplicates ({len(merges)}):[/bold]")
            for a, b, score in merges[:5]:
                console.print(f"  '{a}' ≈ '{b}'  [dim](similarity={score:.0%})[/dim]")


# ── unblock ───────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.argument("concept")
def unblock(vault_str, concept):
    """Re-enable a concept that was blocked after too many rejections."""
    config = _load_config(vault_str)
    db = _load_db(config)

    if not db.is_concept_blocked(concept):
        console.print(f"[yellow]'{concept}' is not blocked.[/yellow]")
        return

    db.unblock_concept(concept)
    count = db.rejection_count(concept)
    console.print(f"[green]'{concept}' unblocked.[/green]")
    console.print(
        f"[dim]{count} rejection(s) remain on record. Next compile will include this concept.[/dim]"
    )


# ── analytics ─────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--export", "export_path", default=None, help="Re-export analytics.jsonl to path")
def analytics(vault_str, export_path):
    """Show token usage, timing, and vault growth analytics."""
    from .analytics import _export_jsonl, get_summary

    config = _load_config(vault_str)
    db = _load_db(config)
    summary = get_summary(db)

    if export_path:
        from pathlib import Path as _Path

        _export_jsonl(db._conn, _Path(export_path))
        console.print(f"[green]Exported to {export_path}[/green]")
        return

    # ── All-time totals ──────────────────────────────────────────────────────
    at = summary.get("all_time") or {}
    total_tokens = (at.get("total_tokens") or 0)
    total_in = (at.get("total_input_tokens") or 0)
    total_out = (at.get("total_output_tokens") or 0)
    total_runs = at.get("total_runs") or 0
    total_docs = at.get("total_docs") or 0
    total_concepts = at.get("total_concepts") or 0
    total_secs = at.get("total_seconds") or 0.0

    totals = Table(title="All-time Totals", show_header=False, box=None, padding=(0, 2))
    totals.add_column("Key", style="bold")
    totals.add_column("Value")
    totals.add_row("Runs", str(total_runs))
    totals.add_row("Docs ingested", f"{total_docs:,}")
    totals.add_row("Concepts compiled", f"{total_concepts:,}")
    totals.add_row("Input tokens", f"{total_in:,}")
    totals.add_row("Output tokens", f"{total_out:,}")
    totals.add_row("Total tokens", f"{total_tokens:,}")
    totals.add_row("Total wall time", f"{total_secs / 3600:.2f} h" if total_secs > 3600 else f"{total_secs:.0f}s")
    console.print(totals)
    console.print()

    # ── Recent runs ──────────────────────────────────────────────────────────
    runs = summary.get("recent_runs") or []
    if runs:
        rt = Table(title="Recent Runs (newest first)", show_header=True)
        rt.add_column("Date", style="dim")
        rt.add_column("Step")
        rt.add_column("Docs", justify="right")
        rt.add_column("Concepts", justify="right")
        rt.add_column("In tok", justify="right")
        rt.add_column("Out tok", justify="right")
        rt.add_column("Time", justify="right")
        rt.add_column("Model", style="dim")
        for r in runs:
            dur_s = (r.get("duration_ms") or 0) / 1000
            dur_str = f"{dur_s:.0f}s" if dur_s < 3600 else f"{dur_s / 3600:.1f}h"
            rt.add_row(
                (r.get("started_at") or "")[:16],
                r.get("pipeline_step") or "",
                str(r.get("docs_processed") or 0),
                str(r.get("concepts_compiled") or 0),
                f'{r.get("total_input_tokens") or 0:,}',
                f'{r.get("total_output_tokens") or 0:,}',
                dur_str,
                r.get("fast_model") or r.get("provider") or "",
            )
        console.print(rt)
        console.print()

    # ── Top 10 slowest docs ──────────────────────────────────────────────────
    slow = summary.get("top_slow_docs") or []
    if slow:
        st = Table(title="Top 10 Slowest Documents", show_header=True)
        st.add_column("Document", style="dim", max_width=50)
        st.add_column("Step")
        st.add_column("Time", justify="right")
        st.add_column("In tok", justify="right")
        st.add_column("Out tok", justify="right")
        st.add_column("Chunks", justify="right")
        for d in slow:
            dur_s = (d.get("duration_ms") or 0) / 1000
            st.add_row(
                (d.get("doc_path") or "").split("/")[-1],
                d.get("pipeline_step") or "",
                f"{dur_s:.1f}s",
                f'{d.get("input_tokens") or 0:,}',
                f'{d.get("output_tokens") or 0:,}',
                str(d.get("chunk_count") or 1),
            )
        console.print(st)
        console.print()

    # ── Efficiency trend ─────────────────────────────────────────────────────
    trend = summary.get("efficiency_trend") or []
    if len(trend) > 1:
        et = Table(title="Efficiency Trend (newest first)", show_header=True)
        et.add_column("Date", style="dim")
        et.add_column("Step")
        et.add_column("Tok/doc", justify="right")
        et.add_column("ms/doc", justify="right")
        for r in trend:
            n = (r.get("docs_processed") or 0) + (r.get("concepts_compiled") or 0)
            if n == 0:
                continue
            tok_per = ((r.get("total_input_tokens") or 0) + (r.get("total_output_tokens") or 0)) // n
            ms_per = (r.get("duration_ms") or 0) // n
            et.add_row(
                (r.get("started_at") or "")[:16],
                r.get("pipeline_step") or "",
                f"{tok_per:,}",
                f"{ms_per:,}",
            )
        console.print(et)
        console.print()

    # ── Machines ─────────────────────────────────────────────────────────────
    machines = summary.get("machines") or []
    if machines:
        mt = Table(title="Known Machines", show_header=True)
        mt.add_column("Host")
        mt.add_column("CPU")
        mt.add_column("Cores", justify="right")
        mt.add_column("RAM", justify="right")
        mt.add_column("GPU")
        mt.add_column("Last seen", style="dim")
        for m in machines:
            mt.add_row(
                m.get("hostname") or "",
                (m.get("cpu_model") or "")[:30],
                str(m.get("cpu_cores") or ""),
                f'{m.get("ram_gb") or 0:.0f} GB',
                m.get("gpu_model") or "—",
                (m.get("last_seen_at") or "")[:16],
            )
        console.print(mt)
