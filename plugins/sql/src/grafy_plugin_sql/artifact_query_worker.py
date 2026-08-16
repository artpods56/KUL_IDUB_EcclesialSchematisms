import resource
import sys


CPU_SECONDS = 30
ADDRESS_SPACE_BYTES = 2 * 1_024 * 1_024 * 1_024
OPEN_FILES = 64


def _set_limit(limit: int, soft: int, hard: int) -> None:
    _current_soft, current_hard = resource.getrlimit(limit)
    effective_hard = hard
    if current_hard != resource.RLIM_INFINITY:
        effective_hard = min(effective_hard, current_hard)
    resource.setrlimit(limit, (min(soft, effective_hard), effective_hard))


def _apply_resource_limits() -> None:
    _set_limit(resource.RLIMIT_CORE, 0, 0)
    _set_limit(resource.RLIMIT_CPU, CPU_SECONDS, CPU_SECONDS + 1)
    _set_limit(resource.RLIMIT_FSIZE, 0, 0)
    _set_limit(resource.RLIMIT_NOFILE, OPEN_FILES, OPEN_FILES)
    if sys.platform != "darwin" and hasattr(resource, "RLIMIT_AS"):
        _set_limit(
            resource.RLIMIT_AS,
            ADDRESS_SPACE_BYTES,
            ADDRESS_SPACE_BYTES,
        )


def main() -> int:
    _apply_resource_limits()
    from grafy_plugin_sql.artifact_query_runtime import run_worker

    return run_worker()


if __name__ == "__main__":
    sys.exit(main())
