"""
Keyword matcher for sensitive API/function names.
Supports case-insensitive matching and '*' wildcard anywhere (e.g., "*RC6*").
If a node has attribute 'api' or 'name', we check both.
"""
from __future__ import annotations
from typing import Iterable, Optional
import re

def _to_regex(pat: str) -> re.Pattern:
    s = pat.strip()
    # Escape regex then bring back wildcard
    s = re.escape(s).replace(r"\*", ".*")
    return re.compile(rf"^{s}$", re.IGNORECASE)

def compile_patterns(keywords: Iterable[str]):
    return [_to_regex(k) for k in keywords]

def is_sensitive(node_api_or_name: Optional[str], compiled):
    if not node_api_or_name:
        return False
    for rgx in compiled:
        if rgx.search(str(node_api_or_name)):
            return True
    return False
