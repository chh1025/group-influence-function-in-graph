"""Process-wide compatibility patches for this repository."""


def _patch_submitit_entrypoint_compat():
    # submitit<=1.5 may assume metadata.entry_points()["submitit"] exists.
    # On some Python environments this raises KeyError.
    try:
        from importlib import metadata
        import submitit.core.plugins as submitit_plugins
    except Exception:
        return

    original_iter = getattr(submitit_plugins, "_iter_submitit_entrypoints", None)
    if original_iter is None or getattr(original_iter, "_eif_patched", False):
        return

    def _safe_iter_submitit_entrypoints():
        eps = metadata.entry_points()
        if hasattr(eps, "select"):
            return eps.select(group="submitit")
        if hasattr(eps, "get"):
            return eps.get("submitit", [])
        try:
            return metadata.entry_points()["submitit"]
        except Exception:
            return [ep for ep in eps if getattr(ep, "group", None) == "submitit"]

    _safe_iter_submitit_entrypoints._eif_patched = True
    submitit_plugins._iter_submitit_entrypoints = _safe_iter_submitit_entrypoints

    for cache_fn_name in ("_get_plugins", "get_executors", "get_job_environments"):
        cache_fn = getattr(submitit_plugins, cache_fn_name, None)
        if cache_fn is not None and hasattr(cache_fn, "cache_clear"):
            cache_fn.cache_clear()


_patch_submitit_entrypoint_compat()
