"""Metrics and artifact registry (P11).

The registry is the single file-based source of truth for every published
numeric metric: `experiments/registry.jsonl` holds one JSON object per
line, one per published run. Each entry records its run ID, the artifact
that carries the numbers and its SHA-256, the input manifest with a
per-file hash, the temporal split, a reproduction command, and one row
per metric with its unit, estimator, comparator, number of origins,
uncertainty and caveats.

Validation is strict on purpose: a metric must resolve to a value inside
its hash-addressed artifact (`artifact_key`, a dotted path such as
`forecast.models[2].holdout_mae`, or a two-key list whose difference is
the value), so a published number can never dangle. A missing artifact,
a changed hash, or a value that no longer matches the artifact all fail
`validate_registry`, and `scripts/verify_frozen_outputs.py` runs that
check on every invocation.

An entry's `artifact_sha256` is normally the SHA-256 of the artifact
file. The frozen report is the exception: a clean rebuild rewrites its
build annotations (the recording git revision), so its raw bytes change
while its content stays identical. Entries for it set
`"artifact_hash_mode": "frozen"`, and the hash is taken over the
canonical frozen bytes (`wattwise.decision_analysis.frozen_report_bytes`),
which are stable across rebuilds by design.
"""

import hashlib
import json
import re
from pathlib import Path

REGISTRY_REL_PATH = "experiments/registry.jsonl"
RUN_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")

ENTRY_FIELDS = {
    "run_id", "title", "artifact", "artifact_sha256", "inputs",
    "temporal_split", "metrics", "reproduction_command",
}
OPTIONAL_ENTRY_FIELDS = {"decision", "artifact_hash_mode"}
HASH_MODES = {"raw", "frozen"}
METRIC_FIELDS = {
    "name", "value", "unit", "estimator", "comparator", "n_origins",
    "uncertainty", "caveats", "artifact_key",
}
_KEY_PART_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(\[\d+\])*")


class RegistryError(ValueError):
    """The registry file is missing, malformed or fails validation."""


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _is_relative(path):
    return not Path(path).is_absolute() and ".." not in Path(path).parts


def _artifact_hash(path, mode):
    """Hash an artifact file, raw or over its canonical frozen bytes."""
    if mode == "raw":
        return _sha256_file(path)
    if mode == "frozen":
        from wattwise.decision_analysis import frozen_report_bytes
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        return hashlib.sha256(frozen_report_bytes(artifact)).hexdigest()
    raise RegistryError(f"unknown artifact_hash_mode: {mode!r}")


def _resolve_key(artifact, key):
    """Resolve a dotted path with [n] indices into a parsed JSON value."""
    current = artifact
    for part in key.split("."):
        match = _KEY_PART_RE.fullmatch(part)
        if match is None:
            raise RegistryError(f"unparsable artifact key {key!r}")
        current = current[match.group(1)]
        for index in re.findall(r"\[(\d+)\]", part):
            current = current[int(index)]
    return current


def _tolerance(value):
    """Half a unit in the last place of the registry value as written."""
    text = repr(value)
    if "." in text:
        decimals = len(text.rsplit(".", 1)[1])
        return 0.5 * 10 ** (-decimals)
    return 0.5


def load_registry(root, path=None):
    """Load the registry entries from `root`, raising RegistryError."""
    root = Path(root)
    registry_path = Path(path) if path is not None else root / REGISTRY_REL_PATH
    if not registry_path.exists():
        raise RegistryError(f"registry file missing: {registry_path}")
    entries = []
    for lineno, line in enumerate(
            registry_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RegistryError(
                f"{registry_path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(entry, dict):
            raise RegistryError(
                f"{registry_path}:{lineno}: entry is not a JSON object")
        entries.append(entry)
    if not entries:
        raise RegistryError(f"registry file is empty: {registry_path}")
    return entries


def _check_metric(entry, metric, artifact, label, errors):
    if not isinstance(metric, dict):
        errors.append(f"{label}: metric is not a JSON object")
        return
    missing = sorted(METRIC_FIELDS - set(metric))
    if missing:
        errors.append(f"{label}: metric missing fields: {', '.join(missing)}")
        return
    name = metric["name"]
    if not isinstance(name, str) or not name:
        errors.append(f"{label}: metric 'name' must be a non-empty string")
    value = metric["value"]
    if value is not None and not isinstance(value, (int, float)):
        errors.append(f"{label}: metric {name!r} 'value' must be a number or null")
    for field in ("unit", "estimator", "comparator", "uncertainty"):
        if not isinstance(metric[field], str) or not metric[field]:
            errors.append(
                f"{label}: metric {name!r} {field!r} must be a non-empty string")
    origins = metric["n_origins"]
    if origins is not None and (not isinstance(origins, int) or origins < 1):
        errors.append(
            f"{label}: metric {name!r} 'n_origins' must be a positive integer or null")
    if not isinstance(metric["caveats"], list) or not all(
            isinstance(c, str) for c in metric["caveats"]):
        errors.append(f"{label}: metric {name!r} 'caveats' must be a list of strings")
    key = metric["artifact_key"]
    if isinstance(key, str):
        keys = [key]
        derived = False
    elif (isinstance(key, list) and len(key) == 2
          and all(isinstance(k, str) for k in key)):
        keys = key
        derived = True
    else:
        errors.append(
            f"{label}: metric {name!r} 'artifact_key' must be a key path "
            "or a two-key list for a derived difference")
        return
    if value is None:
        if not metric["caveats"]:
            errors.append(
                f"{label}: metric {name!r} is null and records no caveat "
                "explaining why")
        return
    try:
        resolved = [_resolve_key(artifact, k) for k in keys]
    except (KeyError, IndexError, TypeError) as exc:
        errors.append(
            f"{label}: metric {name!r} artifact key {key!r} does not resolve: {exc}")
        return
    if not all(isinstance(v, (int, float)) for v in resolved):
        errors.append(
            f"{label}: metric {name!r} artifact key {key!r} is not numeric")
        return
    expected = resolved[0] - resolved[1] if derived else resolved[0]
    tolerance = max(_tolerance(value), 1e-9 * abs(expected))
    if abs(value - expected) > tolerance:
        errors.append(
            f"{label}: metric {name!r} value {value!r} does not match "
            f"artifact value {expected!r} for key {key!r}")


def validate_registry(root, path=None):
    """Validate every registry entry against the files under `root`.

    Returns a list of error strings; an empty list means the registry is
    valid. A missing or hash-mismatched artifact or input, a duplicate
    run ID, a missing field, or a metric that does not resolve to its
    artifact value all produce an error.
    """
    root = Path(root)
    try:
        entries = load_registry(root, path)
    except RegistryError as exc:
        return [str(exc)]
    errors = []
    seen = set()
    for position, entry in enumerate(entries, start=1):
        label = entry.get("run_id", f"entry {position}")
        missing = sorted(ENTRY_FIELDS - set(entry))
        if missing:
            errors.append(f"{label}: missing fields: {', '.join(missing)}")
            continue
        unknown = sorted(set(entry) - ENTRY_FIELDS - OPTIONAL_ENTRY_FIELDS)
        if unknown:
            errors.append(f"{label}: unknown fields: {', '.join(unknown)}")
        run_id = entry["run_id"]
        if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
            errors.append(f"{label}: run_id must match {RUN_ID_RE.pattern!r}")
        if run_id in seen:
            errors.append(f"{label}: duplicate run_id")
        seen.add(run_id)
        if not isinstance(entry["title"], str) or not entry["title"]:
            errors.append(f"{label}: 'title' must be a non-empty string")
        command = entry["reproduction_command"]
        if not isinstance(command, str) or not command.startswith("python "):
            errors.append(
                f"{label}: 'reproduction_command' must be a python command")
        split = entry["temporal_split"]
        if not isinstance(split, dict) or not split.get("description"):
            errors.append(
                f"{label}: 'temporal_split' must record a 'description'")
        artifact_rel = entry["artifact"]
        artifact = None
        hash_mode = entry.get("artifact_hash_mode", "raw")
        if hash_mode not in HASH_MODES:
            errors.append(
                f"{label}: unknown artifact_hash_mode: {hash_mode!r}")
        if not isinstance(artifact_rel, str) or not _is_relative(artifact_rel):
            errors.append(f"{label}: 'artifact' must be a relative path")
        else:
            artifact_path = root / artifact_rel
            if not artifact_path.exists():
                errors.append(f"{label}: artifact missing: {artifact_rel}")
            else:
                try:
                    actual_hash = _artifact_hash(artifact_path, hash_mode)
                except (RegistryError, json.JSONDecodeError,
                        UnicodeDecodeError) as exc:
                    errors.append(f"{label}: cannot hash artifact: {exc}")
                    actual_hash = None
                if actual_hash is not None and actual_hash != entry["artifact_sha256"]:
                    errors.append(
                        f"{label}: artifact sha256 mismatch: {artifact_rel}")
                try:
                    artifact = json.loads(
                        artifact_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    errors.append(
                        f"{label}: artifact is not valid JSON: {exc}")
        inputs = entry["inputs"]
        if not isinstance(inputs, list) or not inputs:
            errors.append(f"{label}: 'inputs' must be a non-empty list")
        else:
            for item in inputs:
                if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                    errors.append(
                        f"{label}: each input needs exactly 'path' and 'sha256'")
                    continue
                if not _is_relative(item["path"]):
                    errors.append(
                        f"{label}: input path must be relative: {item['path']}")
                    continue
                input_path = root / item["path"]
                if not input_path.exists():
                    errors.append(f"{label}: input missing: {item['path']}")
                elif _sha256_file(input_path) != item["sha256"]:
                    errors.append(
                        f"{label}: input sha256 mismatch: {item['path']}")
        metrics = entry["metrics"]
        if not isinstance(metrics, list) or not metrics:
            errors.append(f"{label}: 'metrics' must be a non-empty list")
        elif artifact is not None:
            for metric in metrics:
                _check_metric(entry, metric, artifact, label, errors)
    return errors


def check_registry(root, path=None):
    """Load and validate the registry, raising RegistryError on any error."""
    errors = validate_registry(root, path)
    if errors:
        raise RegistryError("registry invalid:\n" + "\n".join(errors))
    return load_registry(root, path)


def _format_value(value):
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def render_run_table(entries):
    """Render the registered runs as a Markdown table for the README."""
    lines = [
        "| Run ID | Result | Artifact | Reproduce |",
        "|---|---|---|---|",
    ]
    for entry in sorted(entries, key=lambda e: e["run_id"]):
        lines.append(
            f"| `{entry['run_id']}` | {entry['title']} "
            f"| `{entry['artifact']}` | `{entry['reproduction_command']}` |")
    return "\n".join(lines)


def render_metric_table(entry):
    """Render one entry's metrics as a Markdown table."""
    lines = [
        "| Metric | Value | Unit | Estimator | Comparator | Origins | Uncertainty |",
        "|---|---|---|---|---|---|---|",
    ]
    for metric in entry["metrics"]:
        origins = metric["n_origins"] if metric["n_origins"] is not None else "n/a"
        lines.append(
            f"| {metric['name']} | {_format_value(metric['value'])} "
            f"| {metric['unit']} | {metric['estimator']} "
            f"| {metric['comparator']} | {origins} | {metric['uncertainty']} |")
    return "\n".join(lines)


def render_registry_section(entries):
    """Render the README block generated from the registry."""
    lines = [
        "<!-- BEGIN REGISTERED RUNS -->",
        "Every published number in the Evidence table traces to a registered "
        "run in `experiments/registry.jsonl`. Each run records its artifact "
        "hash, input hashes, temporal split, and a reproduction command; "
        "`python scripts/verify_frozen_outputs.py` validates the registry "
        "and fails if an artifact is missing or changed.",
        "",
        render_run_table(entries),
        "<!-- END REGISTERED RUNS -->",
    ]
    return "\n".join(lines)
