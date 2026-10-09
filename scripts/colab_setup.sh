#!/usr/bin/env bash
# Colab: venv with the pinned requirements on top of Colab's own torch; writes /content/env.sh.
#   bash scripts/colab_setup.sh   then in every cell:  source /content/env.sh && ...
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV=/content/venv

if [[ ! -x "${VENV}/bin/pip" ]]; then
  python3 -m venv --without-pip --system-site-packages "${VENV}"
  curl -sS https://bootstrap.pypa.io/get-pip.py | "${VENV}/bin/python" - -q
fi
"${VENV}/bin/pip" install -q -r "${ROOT}/requirements.txt" 2>&1 | grep -viE "dependency resolver|requires|incompatible" || true

cat > /content/env.sh <<EOF
source ${VENV}/bin/activate
export PATH=\$PATH:/usr/lib64-nvidia/bin USE_TF=0 TRANSFORMERS_NO_TF=1 TF_CPP_MIN_LOG_LEVEL=3
cd ${ROOT}
EOF
source /content/env.sh
python -W ignore -c "import torch, transformers, trl; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__, '| trl', trl.__version__)"
