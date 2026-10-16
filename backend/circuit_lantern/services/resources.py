import re
import shutil
from pathlib import Path


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (OSError, UnicodeError):
        return None


def _integer(path: Path) -> int | None:
    value = _read(path)
    try:
        number = int(value) if value is not None else -1
        return number if number >= 0 else None
    except ValueError:
        return None


def _inactive_file(directory: Path, kind: str) -> int:
    counters = {}
    for line in (_read(directory / "memory.stat") or "").splitlines():
        fields = line.split()
        if len(fields) == 2:
            try:
                counters[fields[0]] = max(0, int(fields[1]))
            except ValueError:
                continue
    return (
        counters.get("total_inactive_file", counters.get("inactive_file", 0))
        if kind == "cgroup"
        else counters.get("inactive_file", 0)
    )


def _unescape(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _memory_groups(proc_root: Path):
    groups = {}
    for line in (_read(proc_root / "self/cgroup") or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, group = parts
        if not controllers:
            groups["cgroup2"] = Path(_unescape(group))
        elif "memory" in controllers.split(","):
            groups["cgroup"] = Path(_unescape(group))
    for line in (_read(proc_root / "self/mountinfo") or "").splitlines():
        before, separator, after = line.partition(" - ")
        fields, filesystem = before.split(), after.split()
        if not separator or len(fields) < 5 or len(filesystem) < 3:
            continue
        kind = filesystem[0]
        if kind not in groups or (kind == "cgroup" and "memory" not in filesystem[2].split(",")):
            continue
        mount = Path(_unescape(fields[4])).resolve()
        root = Path(_unescape(fields[3]))
        group = groups[kind]
        relatives = []
        if group.is_relative_to(root):
            relatives.append(group.relative_to(root))
        # A cgroup namespace can expose paths relative to the mounted hierarchy.
        if root != Path("/") and group.is_absolute():
            relatives.append(group.relative_to("/"))
        for relative in relatives:
            current = (mount / relative).resolve()
            if current.is_relative_to(mount) and current.is_dir():
                yield kind, mount, current
                break


def memory_budget(total: int, available: int, *, proc_root: Path = Path("/proc")) -> dict:
    effective_total, effective_available = total, available
    constraints = []
    for kind, mount, current in _memory_groups(proc_root):
        directory = current
        while True:
            if kind == "cgroup2":
                limit = _integer(directory / "memory.max")
                used = _integer(directory / "memory.current")
            else:
                if directory != current and _read(directory / "memory.use_hierarchy") == "0":
                    break
                limit = _integer(directory / "memory.limit_in_bytes")
                used = _integer(directory / "memory.usage_in_bytes")
            # Legacy unlimited cgroups use a value close to the signed integer maximum.
            if limit is not None and limit < 1 << 60:
                reclaimable = min(used, _inactive_file(directory, kind)) if used is not None else 0
                constraints.append(
                    {
                        "source": "cgroup-v2" if kind == "cgroup2" else "cgroup-v1",
                        "limit_bytes": limit,
                        "used_bytes": used,
                        "reclaimable_bytes": reclaimable,
                    }
                )
                effective_total = min(effective_total, limit)
                headroom = max(0, limit - used + reclaimable) if used is not None else 0
                effective_available = min(effective_available, headroom)
            if directory == mount:
                break
            directory = directory.parent
    return {
        "ram_total_bytes": effective_total,
        "ram_available_bytes": max(0, min(effective_total, effective_available)),
        "ram_host_total_bytes": total,
        "ram_host_available_bytes": available,
        "ram_scope": "cgroup"
        if effective_total < total or effective_available < available
        else "system",
        "ram_constraints": constraints,
    }


def _disk_at(path: Path):
    existing = path.expanduser().resolve()
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    return shutil.disk_usage(existing)


def disk_budget(data_dir: Path, checkpoint_cache: Path) -> dict:
    cache = _disk_at(checkpoint_cache)
    records = _disk_at(data_dir)
    return {
        "disk_total_bytes": cache.total,
        "disk_free_bytes": cache.free,
        "checkpoint_cache_path": str(checkpoint_cache.expanduser().absolute()),
        "experiment_disk_total_bytes": records.total,
        "experiment_disk_free_bytes": records.free,
    }
