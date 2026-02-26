# Copyright (C) 2026 KiharaLab, Purdue University
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Interactive tool for preparing CryoZeta input JSON files."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    help="Interactively prepare and validate CryoZeta input JSON files.",
    no_args_is_help=True,
)
console = Console()

# Valid characters for each sequence type
VALID_PROTEIN = set("ARNDCQEGHILKMFPSTWYVX")
VALID_DNA = set("AGCTNIXU")
VALID_RNA = set("AGCUNIX")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_fasta(path: str) -> list[tuple[str, str]]:
    """Read sequences from a FASTA file. Returns list of (header, sequence)."""
    records: list[tuple[str, str]] = []
    header = ""
    seq_parts: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header or seq_parts:
                    records.append((header, "".join(seq_parts)))
                header = line[1:].strip()
                seq_parts = []
            elif line:
                seq_parts.append(line)
    if header or seq_parts:
        records.append((header, "".join(seq_parts)))
    return records


def _validate_sequence(seq: str, seq_type: str) -> list[str]:
    """Return list of error messages (empty if valid)."""
    errors: list[str] = []
    if not seq:
        errors.append("Sequence is empty.")
        return errors
    upper = seq.upper()
    if seq_type == "protein":
        invalid = set(upper) - VALID_PROTEIN
        if invalid:
            errors.append(
                f"Invalid protein characters: {', '.join(sorted(invalid))}. "
                f"Valid: {''.join(sorted(VALID_PROTEIN))}"
            )
    elif seq_type == "dna":
        invalid = set(upper) - VALID_DNA
        if invalid:
            errors.append(
                f"Invalid DNA characters: {', '.join(sorted(invalid))}. "
                f"Valid: {''.join(sorted(VALID_DNA))}"
            )
    elif seq_type == "rna":
        invalid = set(upper) - VALID_RNA
        if invalid:
            errors.append(
                f"Invalid RNA characters: {', '.join(sorted(invalid))}. "
                f"Valid: {''.join(sorted(VALID_RNA))}"
            )
    return errors


def _prompt_positive_float(prompt_text: str) -> float:
    """Prompt until a positive float is entered."""
    while True:
        raw = typer.prompt(prompt_text)
        try:
            val = float(raw)
            if val <= 0:
                console.print("[red]  Value must be positive.[/red]")
                continue
            return val
        except ValueError:
            console.print("[red]  Not a valid number.[/red]")


def _prompt_positive_int(prompt_text: str, default: int | None = None) -> int:
    """Prompt until a positive integer is entered."""
    while True:
        raw = typer.prompt(prompt_text, default=str(default) if default else None)
        try:
            val = int(raw)
            if val <= 0:
                console.print("[red]  Value must be a positive integer.[/red]")
                continue
            return val
        except ValueError:
            console.print("[red]  Not a valid integer.[/red]")


def _prompt_file_path(prompt_text: str, must_exist: bool = True) -> str:
    """Prompt for a file path, optionally checking existence."""
    while True:
        raw = typer.prompt(prompt_text).strip()
        if not raw:
            console.print("[red]  Path cannot be empty.[/red]")
            continue
        p = Path(raw).expanduser()
        if must_exist and not p.exists():
            console.print(f"[red]  File not found: {p}[/red]")
            retry = typer.confirm("  Try again?", default=True)
            if not retry:
                return raw  # accept anyway
            continue
        if must_exist:
            console.print(f"[green]  ✓ File exists[/green]")
        return raw


def _prompt_dir_path(prompt_text: str) -> str | None:
    """Prompt for an optional directory path."""
    raw = typer.prompt(prompt_text, default="").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_dir():
        console.print(f"[yellow]  ⚠ Directory not found: {p}[/yellow]")
    else:
        console.print(f"[green]  ✓ Directory exists[/green]")
    return raw


def _get_sequence_interactive(seq_type: str) -> str:
    """Get a sequence by paste or FASTA file."""
    console.print(f"\n  Sequence input method:")
    console.print(f"    [bold]1[/bold] Paste sequence directly")
    console.print(f"    [bold]2[/bold] Load from FASTA file")
    method = typer.prompt("  Choice", default="1").strip()

    if method == "2":
        fasta_path = _prompt_file_path("  FASTA file path")
        try:
            records = _read_fasta(fasta_path)
        except Exception as e:
            console.print(f"[red]  Failed to read FASTA: {e}[/red]")
            console.print("  Falling back to manual input.")
            return _paste_sequence(seq_type)
        if not records:
            console.print("[red]  No sequences found in FASTA file.[/red]")
            return _paste_sequence(seq_type)
        if len(records) == 1:
            header, seq = records[0]
            console.print(f"  Read: {header} ({len(seq)} residues)")
        else:
            console.print(f"  Found {len(records)} sequences:")
            for i, (h, s) in enumerate(records, 1):
                console.print(f"    [{i}] {h} ({len(s)} residues)")
            idx = _prompt_positive_int("  Select sequence number", default=1)
            idx = min(max(idx, 1), len(records))
            header, seq = records[idx - 1]
        seq = re.sub(r"\s+", "", seq).upper()
        errs = _validate_sequence(seq, seq_type)
        if errs:
            for e in errs:
                console.print(f"[red]  ✗ {e}[/red]")
            if not typer.confirm("  Use this sequence anyway?", default=False):
                return _paste_sequence(seq_type)
        else:
            console.print(f"[green]  ✓ Valid {seq_type} sequence ({len(seq)} residues)[/green]")
        return seq
    else:
        return _paste_sequence(seq_type)


def _paste_sequence(seq_type: str) -> str:
    """Get a pasted sequence with validation."""
    while True:
        raw = typer.prompt("  Sequence").strip()
        seq = re.sub(r"\s+", "", raw).upper()
        errs = _validate_sequence(seq, seq_type)
        if errs:
            for e in errs:
                console.print(f"[red]  ✗ {e}[/red]")
            if typer.confirm("  Use this sequence anyway?", default=False):
                return seq
            continue
        console.print(f"[green]  ✓ Valid {seq_type} sequence ({len(seq)} residues)[/green]")
        return seq


def _build_sequence_entry() -> dict | None:
    """Interactively build one sequence entry."""
    console.print("\n  Sequence type:")
    console.print("    [bold]1[/bold] Protein")
    console.print("    [bold]2[/bold] DNA")
    console.print("    [bold]3[/bold] RNA")
    console.print("    [bold]4[/bold] Ligand")
    console.print("    [bold]5[/bold] Ion")
    choice = typer.prompt("  Choice").strip()

    if choice == "1":
        return _build_protein()
    elif choice == "2":
        return _build_nucleic("dna")
    elif choice == "3":
        return _build_nucleic("rna")
    elif choice == "4":
        return _build_ligand()
    elif choice == "5":
        return _build_ion()
    else:
        console.print(f"[red]  Invalid choice: {choice}[/red]")
        return None


def _build_protein() -> dict:
    seq = _get_sequence_interactive("protein")
    count = _prompt_positive_int("  Number of copies", default=1)
    inner: dict = {"sequence": seq, "count": count}

    # MSA
    msa_dir = _prompt_dir_path("  Precomputed MSA directory (blank to skip)")
    if msa_dir:
        pairing_db = typer.prompt("  Pairing database", default="uniref100").strip()
        inner["msa"] = {
            "precomputed_msa_dir": msa_dir,
            "pairing_db": pairing_db,
        }

    # Modifications
    if typer.confirm("  Add post-translational modifications?", default=False):
        mods = _collect_protein_modifications(len(seq))
        if mods:
            inner["modifications"] = mods

    return {"proteinChain": inner}


def _build_nucleic(na_type: str) -> dict:
    key = "dnaSequence" if na_type == "dna" else "rnaSequence"
    seq = _get_sequence_interactive(na_type)
    count = _prompt_positive_int("  Number of copies", default=1)
    inner: dict = {"sequence": seq, "count": count}

    # MSA (RNA only)
    if na_type == "rna":
        msa_dir = _prompt_dir_path("  Precomputed MSA directory (blank to skip)")
        if msa_dir:
            pairing_db = typer.prompt("  Pairing database", default="uniref100").strip()
            inner["msa"] = {
                "precomputed_msa_dir": msa_dir,
                "pairing_db": pairing_db,
            }

    # Modifications
    if typer.confirm("  Add base modifications?", default=False):
        mods = _collect_nucleic_modifications(len(seq))
        if mods:
            inner["modifications"] = mods

    return {key: inner}


def _build_ligand() -> dict:
    console.print("\n  Ligand format:")
    console.print("    [bold]1[/bold] CCD code (e.g. ATP, HEM)")
    console.print("    [bold]2[/bold] SMILES string")
    console.print("    [bold]3[/bold] File path (.mol, .mol2, .sdf, .pdb)")
    fmt = typer.prompt("  Choice", default="1").strip()

    if fmt == "1":
        code = typer.prompt("  CCD code").strip().upper()
        ligand_str = f"CCD_{code}"
    elif fmt == "2":
        ligand_str = typer.prompt("  SMILES").strip()
    elif fmt == "3":
        path = _prompt_file_path("  Ligand file path")
        ligand_str = f"FILE_{path}"
    else:
        console.print(f"[yellow]  Unknown choice, treating as CCD code.[/yellow]")
        code = typer.prompt("  CCD code").strip().upper()
        ligand_str = f"CCD_{code}"

    count = _prompt_positive_int("  Number of copies", default=1)
    return {"ligand": {"ligand": ligand_str, "count": count}}


def _build_ion() -> dict:
    console.print("  Common ions: NA, MG, ZN, CA, FE, MN, CL, K, CO, CU, NI")
    code = typer.prompt("  Ion CCD code").strip().upper()
    count = _prompt_positive_int("  Number of copies", default=1)
    return {"ion": {"ion": code, "count": count}}


def _collect_protein_modifications(seq_len: int) -> list[dict]:
    mods = []
    while True:
        pos = _prompt_positive_int(f"  Modification position (1-{seq_len})")
        if pos < 1 or pos > seq_len:
            console.print(f"[red]  Position out of range.[/red]")
            continue
        code = typer.prompt("  CCD code for modification (e.g. SEP, TPO)").strip().upper()
        mods.append({"ptmPosition": pos, "ptmType": f"CCD_{code}"})
        console.print(f"[green]  ✓ Added modification CCD_{code} at position {pos}[/green]")
        if not typer.confirm("  Add another modification?", default=False):
            break
    return mods


def _collect_nucleic_modifications(seq_len: int) -> list[dict]:
    mods = []
    while True:
        pos = _prompt_positive_int(f"  Modification position (1-{seq_len})")
        if pos < 1 or pos > seq_len:
            console.print(f"[red]  Position out of range.[/red]")
            continue
        code = typer.prompt("  CCD code for modification").strip().upper()
        mods.append({"basePosition": pos, "modificationType": f"CCD_{code}"})
        console.print(f"[green]  ✓ Added modification CCD_{code} at position {pos}[/green]")
        if not typer.confirm("  Add another modification?", default=False):
            break
    return mods


def _build_covalent_bonds(num_entities: int) -> list[dict]:
    bonds = []
    while True:
        console.print(f"\n  Define a covalent bond (entities 1-{num_entities}):")
        bond: dict = {}
        bond["left_entity"] = _prompt_positive_int("  Left entity index")
        bond["left_position"] = _prompt_positive_int("  Left residue position")
        bond["left_atom"] = typer.prompt("  Left atom name (e.g. C, N, SG)").strip()
        left_copy = typer.prompt("  Left copy (blank for default)", default="").strip()
        if left_copy:
            bond["left_copy"] = int(left_copy)

        bond["right_entity"] = _prompt_positive_int("  Right entity index")
        bond["right_position"] = _prompt_positive_int("  Right residue position")
        bond["right_atom"] = typer.prompt("  Right atom name").strip()
        right_copy = typer.prompt("  Right copy (blank for default)", default="").strip()
        if right_copy:
            bond["right_copy"] = int(right_copy)

        bonds.append(bond)
        console.print(f"[green]  ✓ Added covalent bond[/green]")
        if not typer.confirm("  Add another covalent bond?", default=False):
            break
    return bonds


def _build_entry(entry_num: int, existing_names: set[str]) -> dict:
    """Interactively build one JSON entry."""
    console.print(Panel(f"[bold]Entry {entry_num}[/bold]", expand=False))

    # Name
    while True:
        name = typer.prompt("Name (unique identifier)").strip()
        if not name:
            console.print("[red]  Name cannot be empty.[/red]")
            continue
        if name in existing_names:
            console.print(f"[red]  Name '{name}' already used. Choose a different name.[/red]")
            continue
        break

    # Map path
    map_path = _prompt_file_path("Map file path (.map or .map.gz)")

    # Resolution & contour level
    resolution = _prompt_positive_float("Resolution (Å)")
    contour_level = _prompt_positive_float("Contour level")

    # Sequences
    console.print("\n[bold]── Sequences ──[/bold]")
    sequences: list[dict] = []
    while True:
        if sequences:
            if not typer.confirm("\nAdd another sequence?", default=True):
                break
        else:
            console.print("  (At least one sequence is required)")
        entry = _build_sequence_entry()
        if entry:
            sequences.append(entry)
            console.print(f"[green]  ✓ Sequence {len(sequences)} added[/green]")

    if not sequences:
        console.print("[yellow]  ⚠ No sequences added. Entry will have empty sequences list.[/yellow]")

    result: dict = {
        "name": name,
        "modelSeeds": [],
        "map_path": map_path,
        "resolution": resolution,
        "contour_level": contour_level,
        "sequences": sequences,
    }

    # Covalent bonds (optional)
    if len(sequences) > 1 and typer.confirm(
        "\nAdd covalent bonds between entities?", default=False
    ):
        result["covalent_bonds"] = _build_covalent_bonds(len(sequences))

    return result


def _print_entry_summary(entry: dict, indent: str = "  ") -> list[str]:
    """Validate and summarize one entry. Returns list of errors."""
    errors: list[str] = []
    name = entry.get("name")
    if not name:
        errors.append("'name' is missing")
    if not entry.get("map_path"):
        errors.append("'map_path' is missing")
    if entry.get("resolution") is None:
        errors.append("'resolution' is missing")
    if entry.get("contour_level") is None:
        errors.append("'contour_level' is missing")

    map_exists = Path(entry.get("map_path", "")).expanduser().exists()
    map_status = "[green]exists[/green]" if map_exists else "[yellow]not found[/yellow]"

    console.print(f"{indent}name: [bold]{name}[/bold]")
    console.print(f"{indent}map_path: {entry.get('map_path')} ({map_status})")
    console.print(f"{indent}resolution: {entry.get('resolution')}")
    console.print(f"{indent}contour_level: {entry.get('contour_level')}")

    sequences = entry.get("sequences", [])
    if not sequences:
        errors.append("'sequences' is empty")

    for i, seq_obj in enumerate(sequences, 1):
        if "proteinChain" in seq_obj:
            info = seq_obj["proteinChain"]
            seq = info.get("sequence", "")
            count = info.get("count", "?")
            has_msa = "yes" if info.get("msa") else "no"
            seq_errs = _validate_sequence(seq, "protein")
            mark = "[green]✓[/green]" if not seq_errs else "[red]✗[/red]"
            console.print(
                f"{indent}{mark} Sequence {i}: proteinChain "
                f"({len(seq)} residues, {count} copies, MSA: {has_msa})"
            )
            errors.extend(seq_errs)
        elif "dnaSequence" in seq_obj:
            info = seq_obj["dnaSequence"]
            seq = info.get("sequence", "")
            count = info.get("count", "?")
            seq_errs = _validate_sequence(seq, "dna")
            mark = "[green]✓[/green]" if not seq_errs else "[red]✗[/red]"
            console.print(
                f"{indent}{mark} Sequence {i}: dnaSequence "
                f"({len(seq)} bases, {count} copies)"
            )
            errors.extend(seq_errs)
        elif "rnaSequence" in seq_obj:
            info = seq_obj["rnaSequence"]
            seq = info.get("sequence", "")
            count = info.get("count", "?")
            has_msa = "yes" if info.get("msa") else "no"
            seq_errs = _validate_sequence(seq, "rna")
            mark = "[green]✓[/green]" if not seq_errs else "[red]✗[/red]"
            console.print(
                f"{indent}{mark} Sequence {i}: rnaSequence "
                f"({len(seq)} bases, {count} copies, MSA: {has_msa})"
            )
            errors.extend(seq_errs)
        elif "ligand" in seq_obj:
            info = seq_obj["ligand"]
            console.print(
                f"{indent}[green]✓[/green] Sequence {i}: ligand "
                f"({info.get('ligand', '?')}, {info.get('count', '?')} copies)"
            )
        elif "ion" in seq_obj:
            info = seq_obj["ion"]
            console.print(
                f"{indent}[green]✓[/green] Sequence {i}: ion "
                f"({info.get('ion', '?')}, {info.get('count', '?')} copies)"
            )
        else:
            errors.append(f"Sequence {i}: unknown type (keys: {list(seq_obj.keys())})")
            console.print(f"{indent}[red]✗[/red] Sequence {i}: unknown type")

    return errors


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------


@app.command()
def create(
    output: str = typer.Option(
        "input.json",
        "--output", "-o",
        help="Output JSON file path.",
    ),
):
    """Interactively create a new CryoZeta input JSON file."""
    console.print(
        Panel(
            "[bold]CryoZeta Input JSON Preparation[/bold]\n"
            "Create a new input JSON file interactively.",
            expand=False,
        )
    )

    entries: list[dict] = []
    names: set[str] = set()
    entry_num = 1

    while True:
        entry = _build_entry(entry_num, names)
        entries.append(entry)
        names.add(entry["name"])
        entry_num += 1

        if not typer.confirm("\nAdd another entry?", default=False):
            break

    # Write
    out_path = Path(output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=4)
    console.print(
        f"\n[green]✓ Written {out_path} "
        f"({len(entries)} {'entry' if len(entries) == 1 else 'entries'})[/green]"
    )


@app.command()
def append(
    json_file: str = typer.Argument(help="Existing JSON file to add entries to."),
):
    """Add entries to an existing CryoZeta input JSON file."""
    path = Path(json_file).expanduser()
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    with open(path) as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        console.print("[red]JSON root must be a list.[/red]")
        raise typer.Exit(1)

    names = {e.get("name") for e in entries if e.get("name")}
    console.print(f"Loaded {len(entries)} existing entries from {path}")
    console.print(f"Existing names: {', '.join(sorted(names)) if names else '(none)'}")

    entry_num = len(entries) + 1
    while True:
        entry = _build_entry(entry_num, names)
        entries.append(entry)
        names.add(entry["name"])
        entry_num += 1

        if not typer.confirm("\nAdd another entry?", default=False):
            break

    with open(path, "w") as f:
        json.dump(entries, f, indent=4)
    console.print(f"\n[green]✓ Updated {path} ({len(entries)} entries total)[/green]")


@app.command()
def ui(
    port: int = typer.Option(8501, help="Port for the web UI server."),
):
    """Launch the browser-based JSON preparation UI."""
    from cryozeta.runner.prepare_json_ui import run_server

    console.print(f"[bold]Launching CryoZeta JSON Prep UI on port {port}...[/bold]")
    console.print(f"Open http://localhost:{port} in your browser.")
    console.print(f"For SSH access: ssh -L {port}:localhost:{port} <user>@<host>")
    run_server(port=port)


@app.command()
def validate(
    json_file: str = typer.Argument(help="JSON file to validate."),
):
    """Validate an existing CryoZeta input JSON file."""
    path = Path(json_file).expanduser()
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON: {e}[/red]")
        raise typer.Exit(1)

    if not isinstance(data, list):
        console.print("[red]JSON root must be a list of entries.[/red]")
        raise typer.Exit(1)

    console.print(Panel(f"[bold]Validating {path}[/bold]", expand=False))

    total_errors: list[str] = []
    seen_names: set[str] = set()

    for i, entry in enumerate(data, 1):
        console.print(f"\n[bold]Entry {i}:[/bold]")
        entry_errors = _print_entry_summary(entry)
        name = entry.get("name", "")
        if name in seen_names:
            err = f"Duplicate entry name: '{name}'"
            entry_errors.append(err)
            console.print(f"  [red]✗ {err}[/red]")
        seen_names.add(name)
        total_errors.extend(entry_errors)

    console.print()
    if total_errors:
        console.print(
            f"[red]✗ Found {len(total_errors)} issue(s) "
            f"across {len(data)} entries.[/red]"
        )
        raise typer.Exit(1)
    else:
        console.print(
            f"[green]✓ All {len(data)} "
            f"{'entry' if len(data) == 1 else 'entries'} valid.[/green]"
        )
