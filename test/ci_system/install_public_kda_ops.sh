#!/usr/bin/env bash
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set -euo pipefail

TOKENSPEED_CANN_ROOT="${TOKENSPEED_CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOKENSPEED_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TOKENSPEED_PUBLIC_KDA_BUILD_ROOT="${TOKENSPEED_PUBLIC_KDA_BUILD_ROOT:-${TOKENSPEED_REPO_ROOT}/build/public_kda_ops}"
LOCK_PATH="${TOKENSPEED_REPO_ROOT}/tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/thirdparty/public_kda_ops.lock.json"

if [[ ! -r "${TOKENSPEED_CANN_ROOT}/set_env.sh" ]]; then
    echo "CANN environment script not found: ${TOKENSPEED_CANN_ROOT}/set_env.sh" >&2
    exit 1
fi

# shellcheck disable=SC1091
set +u
source "${TOKENSPEED_CANN_ROOT}/set_env.sh"
set -u

mapfile -t lock_values < <(
    python - "${LOCK_PATH}" <<'PY'
import json
import sys

lock = json.load(open(sys.argv[1]))
print(lock["repository"])
print(lock["commit"])
PY
)

source_dir="${TOKENSPEED_PUBLIC_KDA_SOURCE_DIR:-}"
if [[ -z "${source_dir}" ]]; then
    source_dir="${TOKENSPEED_PUBLIC_KDA_BUILD_ROOT}/source"
    mkdir -p "${TOKENSPEED_PUBLIC_KDA_BUILD_ROOT}"
    if [[ ! -d "${source_dir}/.git" ]]; then
        git clone --filter=blob:none --no-checkout "${lock_values[0]}" "${source_dir}"
    fi
    git -C "${source_dir}" fetch --depth 1 origin "${lock_values[1]}"
    git -C "${source_dir}" checkout --detach --force "${lock_values[1]}"
    git -C "${source_dir}" submodule update --init --depth 1 csrc/third_party/catlass
fi

python "${TOKENSPEED_REPO_ROOT}/tokenspeed-kernel-npu/tools/build_public_kda_ops.py" \
    --source-dir "${source_dir}"
