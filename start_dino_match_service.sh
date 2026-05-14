#!/usr/bin/env bash
set -Eeuo pipefail

# DINO 匹配 FastAPI 服务启动脚本。
# 默认会优先使用项目内的 .venv/venv，未找到时回退到系统 python3/python。

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="${BASE_DIR}/fastapi_dino_match_service.py"
PID_DIR="${BASE_DIR}/run"
LOG_DIR="${BASE_DIR}/logs"
PID_FILE="${PID_FILE:-${PID_DIR}/fastapi_dino_match_service.pid}"
OUT_LOG_FILE="${OUT_LOG_FILE:-${LOG_DIR}/fastapi_dino_match_service.out.log}"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

find_python() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        printf '%s\n' "${PYTHON_BIN}"
        return 0
    fi

    if [[ -x "${BASE_DIR}/.venv/bin/python" ]]; then
        printf '%s\n' "${BASE_DIR}/.venv/bin/python"
        return 0
    fi

    if [[ -x "${BASE_DIR}/venv/bin/python" ]]; then
        printf '%s\n' "${BASE_DIR}/venv/bin/python"
        return 0
    fi

    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi

    command -v python
}

is_service_running() {
    local pid="$1"

    if [[ -z "${pid}" ]] || ! kill -0 "${pid}" >/dev/null 2>&1; then
        return 1
    fi

    # 避免 PID 被复用时误判，只接受命令行中包含当前服务脚本名的进程。
    ps -p "${pid}" -o args= 2>/dev/null | grep -F "fastapi_dino_match_service.py" >/dev/null 2>&1
}

if [[ ! -f "${SERVICE_FILE}" ]]; then
    log "启动失败：未找到服务文件 ${SERVICE_FILE}"
    exit 1
fi

mkdir -p "${PID_DIR}" "${LOG_DIR}"

if [[ -f "${PID_FILE}" ]]; then
    old_pid="$(tr -d '[:space:]' < "${PID_FILE}")"
    if is_service_running "${old_pid}"; then
        log "服务已经在运行，PID=${old_pid}"
        exit 0
    fi

    log "发现过期 PID 文件，正在清理：${PID_FILE}"
    rm -f "${PID_FILE}"
fi

if ! PYTHON_EXECUTABLE="$(find_python)"; then
    log "启动失败：未找到可用的 python/python3，请设置 PYTHON_BIN 后重试"
    exit 1
fi

cd "${BASE_DIR}"
log "正在启动 DINO 匹配服务，日志：${OUT_LOG_FILE}"
PYTHONUNBUFFERED=1 nohup "${PYTHON_EXECUTABLE}" "${SERVICE_FILE}" >> "${OUT_LOG_FILE}" 2>&1 &
service_pid="$!"
printf '%s\n' "${service_pid}" > "${PID_FILE}"

sleep 2
if is_service_running "${service_pid}"; then
    log "服务启动成功，PID=${service_pid}，监听端口默认使用 8003"
else
    log "服务启动失败，请查看日志：${OUT_LOG_FILE}"
    rm -f "${PID_FILE}"
    exit 1
fi
