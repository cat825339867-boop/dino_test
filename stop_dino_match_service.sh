#!/usr/bin/env bash
set -Eeuo pipefail

# DINO 匹配 FastAPI 服务停止脚本。
# 优先按 PID 文件停止，PID 文件缺失时会按服务脚本名查找进程。

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="${BASE_DIR}/run"
PID_FILE="${PID_FILE:-${PID_DIR}/fastapi_dino_match_service.pid}"
STOP_TIMEOUT_SECONDS="${STOP_TIMEOUT_SECONDS:-15}"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

is_service_running() {
    local pid="$1"

    if [[ -z "${pid}" ]] || ! kill -0 "${pid}" >/dev/null 2>&1; then
        return 1
    fi

    # 只匹配当前服务脚本名，避免误停其它 Python 服务。
    ps -p "${pid}" -o args= 2>/dev/null | grep -F "fastapi_dino_match_service.py" >/dev/null 2>&1
}

find_service_pids() {
    pgrep -f "fastapi_dino_match_service.py" 2>/dev/null || true
}

stop_pid() {
    local pid="$1"
    local waited=0

    if ! is_service_running "${pid}"; then
        log "进程不存在或不是目标服务，PID=${pid}"
        return 0
    fi

    log "正在停止服务，PID=${pid}"
    kill "${pid}"

    while is_service_running "${pid}"; do
        if (( waited >= STOP_TIMEOUT_SECONDS )); then
            log "优雅停止超时，强制结束 PID=${pid}"
            kill -9 "${pid}" >/dev/null 2>&1 || true
            break
        fi

        sleep 1
        waited=$((waited + 1))
    done

    if is_service_running "${pid}"; then
        log "停止失败，PID=${pid} 仍在运行"
        return 1
    fi

    log "服务已停止，PID=${pid}"
}

target_pids=()
if [[ -f "${PID_FILE}" ]]; then
    pid_from_file="$(tr -d '[:space:]' < "${PID_FILE}")"
    if [[ -n "${pid_from_file}" ]]; then
        target_pids+=("${pid_from_file}")
    fi
fi

if (( ${#target_pids[@]} == 0 )); then
    while IFS= read -r pid; do
        [[ -n "${pid}" ]] && target_pids+=("${pid}")
    done < <(find_service_pids)
fi

if (( ${#target_pids[@]} == 0 )); then
    log "未发现正在运行的 DINO 匹配服务"
    rm -f "${PID_FILE}"
    exit 0
fi

for pid in "${target_pids[@]}"; do
    stop_pid "${pid}"
done

rm -f "${PID_FILE}"
