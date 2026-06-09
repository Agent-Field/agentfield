"""OpenRouter attribution helpers.

This module only deals with OpenRouter app attribution metadata. It must not
read or write AgentField control-plane auth headers such as X-API-Key.
"""

from __future__ import annotations

import os
from typing import Dict, Mapping, MutableMapping, Optional

DEFAULT_OPENROUTER_SITE_URL = "https://agentfield.ai"
DEFAULT_OPENROUTER_APP_NAME = "AgentField AI"

_FALSE_VALUES = {"0", "false", "no", "off"}


def attribution_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    source = env if env is not None else os.environ
    value = str(source.get("AGENTFIELD_OPENROUTER_ATTRIBUTION", "")).strip().lower()
    return value not in _FALSE_VALUES


def is_openrouter_request(
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    base_url: Optional[str] = None,
) -> bool:
    model_value = (model or "").strip().lower()
    if model_value.startswith("openrouter/"):
        return True

    for value in (api_base, base_url):
        if value and "openrouter.ai" in value.lower():
            return True
    return False


def resolve_attribution(
    *,
    site_url: Optional[str] = None,
    app_name: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> tuple[str, str]:
    source = env if env is not None else os.environ
    resolved_site = (
        _clean(site_url)
        or _clean(source.get("AGENTFIELD_OPENROUTER_SITE_URL"))
        or _clean(source.get("OR_SITE_URL"))
        or DEFAULT_OPENROUTER_SITE_URL
    )
    resolved_name = (
        _clean(app_name)
        or _clean(source.get("AGENTFIELD_OPENROUTER_APP_NAME"))
        or _clean(source.get("OR_APP_NAME"))
        or DEFAULT_OPENROUTER_APP_NAME
    )
    return resolved_site, resolved_name


def attribution_headers(
    *,
    site_url: Optional[str] = None,
    app_name: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    if not attribution_enabled(env):
        return {}

    resolved_site, resolved_name = resolve_attribution(
        site_url=site_url, app_name=app_name, env=env
    )
    headers: Dict[str, str] = {}
    if resolved_site:
        headers["HTTP-Referer"] = resolved_site
    if resolved_name:
        headers["X-OpenRouter-Title"] = resolved_name
        headers["X-Title"] = resolved_name
    return headers


def merge_attribution_headers(
    existing: Optional[Mapping[str, str]] = None,
    *,
    site_url: Optional[str] = None,
    app_name: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    merged = dict(existing or {})
    lower_keys = {key.lower() for key in merged}
    for key, value in attribution_headers(
        site_url=site_url, app_name=app_name, env=env
    ).items():
        if key.lower() not in lower_keys:
            merged[key] = value
            lower_keys.add(key.lower())
    return merged


def apply_litellm_attribution(
    params: MutableMapping[str, object],
    *,
    site_url: Optional[str] = None,
    app_name: Optional[str] = None,
) -> None:
    if not is_openrouter_request(
        model=str(params.get("model") or ""),
        api_base=str(params.get("api_base") or ""),
        base_url=str(params.get("base_url") or ""),
    ):
        return

    params["headers"] = merge_attribution_headers(
        _string_dict(params.get("headers")),
        site_url=site_url,
        app_name=app_name,
    )
    params["extra_headers"] = merge_attribution_headers(
        _string_dict(params.get("extra_headers")),
        site_url=site_url,
        app_name=app_name,
    )


def attribution_env(env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    source = env if env is not None else os.environ
    if not attribution_enabled(source):
        return {}

    site_url, app_name = resolve_attribution(env=source)
    result: Dict[str, str] = {}
    if site_url:
        result["AGENTFIELD_OPENROUTER_SITE_URL"] = site_url
        result["OR_SITE_URL"] = site_url
    if app_name:
        result["AGENTFIELD_OPENROUTER_APP_NAME"] = app_name
        result["OR_APP_NAME"] = app_name
    return result


def apply_subprocess_env(env: MutableMapping[str, str]) -> None:
    if not attribution_enabled(env):
        for key in (
            "AGENTFIELD_OPENROUTER_SITE_URL",
            "AGENTFIELD_OPENROUTER_APP_NAME",
            "OR_SITE_URL",
            "OR_APP_NAME",
        ):
            env.pop(key, None)
        return

    for key, value in attribution_env(env).items():
        env.setdefault(key, value)


def _clean(value: Optional[str]) -> str:
    return str(value or "").strip()


def _string_dict(value: object) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    result: Dict[str, str] = {}
    for key, item in value.items():
        if isinstance(key, str) and item is not None:
            result[key] = str(item)
    return result
