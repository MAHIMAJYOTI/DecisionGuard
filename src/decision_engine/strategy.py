"""Load a strategy YAML and reject it if it does not match the schema.

The schema is the contract. A typo, a missing owner, or a decision spelled
wrong fails here, before any application is scored.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from src.data_gen.config import project_root

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class StrategyError(ValueError):
    """The YAML is missing, duplicated, or does not match the schema."""


@dataclass(frozen=True)
class Comparison:
    field: str
    op: str
    value: float


@dataclass(frozen=True)
class NamedCheck:
    field: str
    decision: str
    reason: str


@dataclass(frozen=True)
class PolicyRule:
    field: str
    op: str
    value: float
    decision: str
    reason: str


@dataclass(frozen=True)
class FraudRule:
    decision: str
    reason: str
    when: tuple[Comparison, ...]


@dataclass(frozen=True)
class ThinFileBand:
    field: str
    op: str
    value: float
    approve_max_pd: float
    refer_max_pd: float


@dataclass(frozen=True)
class Strategy:
    version: str
    effective_date: str
    owner: str
    description: str
    kind: str
    model_version: str
    data_checks: tuple[NamedCheck, ...]
    policy_rules: tuple[PolicyRule, ...]
    fraud_rules: tuple[FraudRule, ...]
    approve_max_pd: float | None
    refer_max_pd: float | None
    thin_file: ThinFileBand | None
    challenge_min_pd: float | None
    block_min_pd: float | None
    risk_bands: tuple[dict[str, float | str], ...]
    limit_rules: tuple[dict[str, float | str], ...]
    path: Path


def load_strategy(version: str) -> Strategy:
    """Find strategies/<file>.yaml whose version field equals `version`."""

    folder = project_root() / "strategies"
    matches = [path for path in sorted(folder.glob("*.yaml")) if _version_of(path) == version]
    if not matches:
        raise StrategyError(f"No strategy with version {version!r} in {folder}.")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise StrategyError(f"More than one file has version {version!r}: {names}.")
    return load_strategy_file(matches[0])


def load_strategy_file(path: Path) -> Strategy:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise StrategyError(f"{path} must be a YAML mapping.")
    # YAML turns an unquoted 2026-10-01 into a date. The schema wants the text.
    effective = document.get("effective_date")
    if isinstance(effective, datetime):
        document["effective_date"] = effective.date().isoformat()
    elif isinstance(effective, date):
        document["effective_date"] = effective.isoformat()
    _validate_schema(path, document)
    _validate_dates_and_bands(path, document)
    return _build(path, document)


def _version_of(path: Path) -> str | None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(document, dict):
        version = document.get("version")
        if isinstance(version, str):
            return version
    return None


def _validate_schema(path: Path, document: dict) -> None:
    kind = document.get("kind")
    if kind not in {"credit", "fraud"}:
        raise StrategyError(f"{path} kind must be 'credit' or 'fraud' before the schema can be chosen.")
    schema_path = project_root() / "strategies" / "schema" / f"{kind}_strategy.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.path))
    if not errors:
        return
    lines = [f"{path.name} does not match the {kind} strategy schema:"]
    for error in errors:
        where = ".".join(str(part) for part in error.path) or "(file)"
        lines.append(f"  {where}: {error.message}")
    raise StrategyError("\n".join(lines))


def _validate_dates_and_bands(path: Path, document: dict) -> None:
    effective = document["effective_date"]
    if not isinstance(effective, str) or not _DATE.match(effective):
        raise StrategyError(f"{path.name} effective_date must be YYYY-MM-DD.")
    try:
        datetime.strptime(effective, "%Y-%m-%d")
    except ValueError as exc:
        raise StrategyError(f"{path.name} effective_date is not a real date: {effective}.") from exc

    if document["kind"] == "credit":
        bands = document["score_bands"]
        approve = float(bands["approve_max_pd"])
        refer = float(bands["refer_max_pd"])
        thin = bands["thin_file"]
        thin_approve = float(thin["approve_max_pd"])
        thin_refer = float(thin["refer_max_pd"])
        if not approve < refer:
            raise StrategyError(f"{path.name} approve_max_pd must be below refer_max_pd.")
        if not thin_approve < thin_refer:
            raise StrategyError(f"{path.name} thin-file approve_max_pd must be below its refer_max_pd.")
        if not thin_approve < approve or not thin_refer < refer:
            raise StrategyError(
                f"{path.name} thin-file bands must be tighter than the ordinary bands."
            )
        band_names = [item["name"] for item in document["risk_bands"]]
        if len(band_names) != len(set(band_names)):
            raise StrategyError(f"{path.name} repeats a risk band name.")
        for rule in document["limit_rules"]:
            if rule["risk_band"] not in band_names:
                raise StrategyError(
                    f"{path.name} limit rule {rule['risk_band']!r} is not a risk band."
                )
        return

    challenge = float(document["score_bands"]["challenge_min_pd"])
    block = float(document["score_bands"]["block_min_pd"])
    if not challenge < block:
        raise StrategyError(f"{path.name} challenge_min_pd must be below block_min_pd.")


def _build(path: Path, document: dict) -> Strategy:
    checks = tuple(
        NamedCheck(item["field"], item["decision"], item["reason"]) for item in document["data_checks"]
    )
    if document["kind"] == "credit":
        bands = document["score_bands"]
        thin = bands["thin_file"]
        return Strategy(
            version=document["version"],
            effective_date=document["effective_date"],
            owner=document["owner"],
            description=document["description"].strip(),
            kind="credit",
            model_version=document["model_version"],
            data_checks=checks,
            policy_rules=tuple(
                PolicyRule(item["field"], item["op"], float(item["value"]), item["decision"], item["reason"])
                for item in document["policy_rules"]
            ),
            fraud_rules=(),
            approve_max_pd=float(bands["approve_max_pd"]),
            refer_max_pd=float(bands["refer_max_pd"]),
            thin_file=ThinFileBand(
                thin["field"],
                thin["op"],
                float(thin["value"]),
                float(thin["approve_max_pd"]),
                float(thin["refer_max_pd"]),
            ),
            challenge_min_pd=None,
            block_min_pd=None,
            risk_bands=tuple(document["risk_bands"]),
            limit_rules=tuple(document["limit_rules"]),
            path=path,
        )
    return Strategy(
        version=document["version"],
        effective_date=document["effective_date"],
        owner=document["owner"],
        description=document["description"].strip(),
        kind="fraud",
        model_version=document["model_version"],
        data_checks=checks,
        policy_rules=(),
        fraud_rules=tuple(
            FraudRule(
                item["decision"],
                item["reason"],
                tuple(Comparison(part["field"], part["op"], float(part["value"])) for part in item["when"]),
            )
            for item in document["rules"]
        ),
        approve_max_pd=None,
        refer_max_pd=None,
        thin_file=None,
        challenge_min_pd=float(document["score_bands"]["challenge_min_pd"]),
        block_min_pd=float(document["score_bands"]["block_min_pd"]),
        risk_bands=(),
        limit_rules=(),
        path=path,
    )
