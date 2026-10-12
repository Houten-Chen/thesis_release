"""Alphabet definitions transcribed from the five uploaded script variants."""

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("alphabets.json")
ALPHABETS = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def get_alphabet(name):
    if name not in ALPHABETS:
        raise ValueError(
            f"Unknown alphabet {name!r}; choose from {', '.join(ALPHABETS)}"
        )
    config = dict(ALPHABETS[name])
    residues = "".join(config["groups"])
    if sorted(residues) != sorted("ACDEFGHIKLMNPQRSTVWY"):
        raise ValueError(f"{name}: groups must partition the 20 standard amino acids")
    if len(config["H1"]) != len(config["groups"]) or len(config["H2"]) != len(
        config["groups"]
    ):
        raise ValueError(f"{name}: H1/H2 lengths do not match group count")
    if config["unknown"] != str(len(config["groups"]) + 1):
        raise ValueError(f"{name}: unexpected placeholder code")
    config["alphabet"] = name
    return config


def translation_map(config):
    return {
        residue: str(i)
        for i, group in enumerate(config["groups"], start=1)
        for residue in group
    }
