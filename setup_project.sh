#!/bin/bash
# =============================================================================
#  PY-BRAIN — Full Automatic Setup Script
#  Apple Silicon Mac (M1/M2/M3/M4) — macOS
# =============================================================================
#
#  What this script does automatically:
#    1. Checks/installs Homebrew
#    2. Checks/installs Python 3.9 via Homebrew
#    3. Creates the .venv virtual environment
#    4. Installs ALL required Python packages
#    5. Creates the full project directory structure
#    6. Verifies every package is working
#    7. Prints PyCharm setup instructions
#
#  Usage (run ONCE from Terminal):
#    cd ~/documents/PY-BRAIN
#    chmod +x setup_project.sh
#    ./setup_project.sh
#
#  To RESET everything and start fresh:
#    rm -rf .venv && ./setup_project.sh
# =============================================================================

# ── Colours ──────────────────────────────────────────────────────────────────
R="\033[0m"; BOLD="\033[1m"
CY="\033[96m"; GR="\033[92m"; YL="\033[93m"; RD="\033[91m"; GY="\033[90m"

h()  { echo -e "\n${CY}${BOLD}══════════════════════════════════════════════════${R}"; \
       echo -e "${CY}${BOLD}  $1${R}"; \
       echo -e "${CY}${BOLD}══════════════════════════════════════════════════${R}"; }
ok() { echo -e "  ${GR}✅${R}  $1"; }
info(){ echo -e "  ${CY}ℹ${R}  $1"; }
warn(){ echo -e "  ${YL}⚠️ ${R}  $1"; }
err() { echo -e "  ${RD}❌${R}  $1"; }
step(){ echo -e "\n${BOLD}[$1/$TOTAL_STEPS] $2${R}"; }

TOTAL_STEPS=7

# ── Project root = folder containing this script ─────────────────────────────
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$PROJECT_DIR/.venv"
LOG_FILE="$PROJECT_DIR/logs/setup_$(date +%Y%m%d_%H%M%S).log"

# Create logs dir early
mkdir -p "$PROJECT_DIR/logs"

# Redirect all output to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo ""
echo -e "${CY}${BOLD}"
echo "  ██████╗ ██╗   ██╗      ██████╗ ██████╗  █████╗ ██╗███╗   ██╗"
echo "  ██╔══██╗╚██╗ ██╔╝      ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║"
echo "  ██████╔╝ ╚████╔╝ █████╗██████╔╝██████╔╝███████║██║██╔██╗ ██║"
echo "  ██╔═══╝   ╚██╔╝  ╚════╝██╔══██╗██╔══██╗██╔══██║██║██║╚██╗██║"
echo "  ██║        ██║         ██████╔╝██║  ██║██║  ██║██║██║ ╚████║"
echo "  ╚═╝        ╚═╝         ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝"
echo -e "${R}"
echo -e "  ${GY}Brain Tumour AI Analysis Pipeline — Full Setup${R}"
echo -e "  ${GY}Project root: $PROJECT_DIR${R}"
echo ""

# =============================================================================
# STEP 1 — HOMEBREW
# =============================================================================
step 1 "Homebrew"

if command -v brew &>/dev/null; then
    ok "Homebrew already installed: $(brew --version | head -1)"
else
    info "Homebrew not found — installing..."
    info "This may ask for your Mac password."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add Homebrew to PATH for Apple Silicon
    if [ -f "/opt/homebrew/bin/brew" ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
        # Persist to shell profile
        PROFILE="$HOME/.zprofile"
        if ! grep -q "homebrew" "$PROFILE" 2>/dev/null; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$PROFILE"
            info "Added Homebrew to ~/.zprofile"
        fi
    fi

    if command -v brew &>/dev/null; then
        ok "Homebrew installed successfully"
    else
        err "Homebrew installation failed."
        err "Install manually: https://brew.sh"
        exit 1
    fi
fi

# Ensure Homebrew is in PATH (Apple Silicon path)
if [ -f "/opt/homebrew/bin/brew" ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# =============================================================================
# STEP 2 — PYTHON 3.9
# =============================================================================
step 2 "Python 3.9"

PYTHON_BIN=""

# Check for Python 3.9 in preferred locations
for PY_PATH in \
    "/opt/homebrew/bin/python3.9" \
    "/usr/local/bin/python3.9" \
    "$(brew --prefix 2>/dev/null)/bin/python3.9"; do
    if [ -f "$PY_PATH" ]; then
        PYTHON_BIN="$PY_PATH"
        break
    fi
done

if [ -n "$PYTHON_BIN" ]; then
    ok "Python 3.9 found: $($PYTHON_BIN --version) at $PYTHON_BIN"
else
    info "Python 3.9 not found — installing via Homebrew..."
    brew install python@3.9

    # Find it after install
    for PY_PATH in \
        "/opt/homebrew/bin/python3.9" \
        "/opt/homebrew/opt/python@3.9/bin/python3.9" \
        "$(brew --prefix python@3.9 2>/dev/null)/bin/python3.9"; do
        if [ -f "$PY_PATH" ]; then
            PYTHON_BIN="$PY_PATH"
            break
        fi
    done

    if [ -n "$PYTHON_BIN" ]; then
        ok "Python 3.9 installed: $($PYTHON_BIN --version)"
    else
        err "Python 3.9 install failed. Trying fallback versions..."
        for PY in python3.10 python3.11 python3.12 python3; do
            if command -v $PY &>/dev/null; then
                PYTHON_BIN=$(which $PY)
                warn "Using $PY instead: $($PYTHON_BIN --version)"
                break
            fi
        done
        if [ -z "$PYTHON_BIN" ]; then
            err "No usable Python found. Install Python 3.9 manually."
            exit 1
        fi
    fi
fi

PY_VERSION=$($PYTHON_BIN --version 2>&1)
info "Using: $PY_VERSION  at  $PYTHON_BIN"

# =============================================================================
# STEP 3 — DIRECTORY STRUCTURE
# =============================================================================
step 3 "Project directory structure"

# All directories the pipeline needs
DIRS=(
    "$PROJECT_DIR/nifti/monai_ready"           # BraTS NIfTI
    "$PROJECT_DIR/nifti/extra_sequences"       # extra NIfTI
    "$PROJECT_DIR/nifti/ct_converted"          # CT work dir
    "$PROJECT_DIR/results"                     # analysis outputs
    "$PROJECT_DIR/models/brats_bundle"         # model weights
    "$PROJECT_DIR/scripts"                     # pipeline scripts
    "$PROJECT_DIR/config"                      # config.py
    "$PROJECT_DIR/notebooks"                   # Jupyter
    "$PROJECT_DIR/tests"                       # unit tests
    "$PROJECT_DIR/docs"                        # documentation
    "$PROJECT_DIR/logs"                        # run logs
)

for DIR in "${DIRS[@]}"; do
    mkdir -p "$DIR"
done

ok "Directory structure created"
echo ""
echo -e "  ${GY}$PROJECT_DIR/${R}"
echo -e "  ${GY}├── nifti/              ← auto-filled by pipeline${R}"
echo -e "  ${GY}├── models/             ← AI model weights${R}"
echo -e "  ${GY}├── results/            ← all analysis outputs${R}"
echo -e "  ${GY}├── scripts/            ← the 9 pipeline scripts${R}"
echo -e "  ${GY}├── config/             ← config.py${R}"
echo -e "  ${GY}└── .venv/              ← Python virtual environment${R}"

# =============================================================================
# STEP 4 — VIRTUAL ENVIRONMENT
# =============================================================================
step 4 "Virtual environment (.venv)"

if [ -d "$VENV_DIR" ]; then
    warn ".venv already exists"
    read -p "  Recreate from scratch? [y/N] " RECREATE
    if [[ "$RECREATE" =~ ^[Yy]$ ]]; then
        info "Removing old .venv..."
        rm -rf "$VENV_DIR"
    else
        info "Keeping existing .venv"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    info "Creating .venv with $PY_VERSION..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"

    if [ -d "$VENV_DIR" ]; then
        ok "Created: $VENV_DIR"
    else
        err "Failed to create .venv"
        exit 1
    fi
fi

# Activate
source "$VENV_DIR/bin/activate"

if [[ "$VIRTUAL_ENV" == "$VENV_DIR" ]]; then
    ok "Activated: $(which python3)  |  $(python3 --version)"
else
    err "Failed to activate .venv"
    exit 1
fi

# =============================================================================
# STEP 5 — PIP UPGRADE
# =============================================================================
step 5 "Upgrading pip / wheel / setuptools"

pip install --upgrade pip wheel setuptools --quiet
ok "pip $(pip --version | cut -d' ' -f2)  |  ready"

# =============================================================================
# STEP 6 — INSTALL PACKAGES
# =============================================================================
step 6 "Installing Python packages"

info "This takes 5-15 minutes on first run. Grab a coffee ☕"
echo ""

# Helper — install with retry and fallback
install_pkg() {
    local NAME="$1"
    shift
    echo -e "  ${CY}→${R} Installing $NAME..."
    if pip install "$@" --quiet 2>/dev/null; then
        ok "$NAME"
        return 0
    else
        warn "$NAME — retrying without --quiet for error details..."
        pip install "$@" 2>&1 | tail -5
        return 1
    fi
}

# ── PyTorch (Apple Silicon CPU wheel — MPS is built-in to torch) ─────────────
install_pkg "PyTorch + TorchVision" \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# ── MONAI ────────────────────────────────────────────────────────────────────
install_pkg "MONAI (full)" "monai[all]>=1.3"

# ── Core scientific stack ─────────────────────────────────────────────────────
install_pkg "NumPy"        "numpy>=1.24,<2.0"
install_pkg "SciPy"        "scipy>=1.10"
install_pkg "scikit-image" "scikit-image>=0.21"
install_pkg "scikit-learn" "scikit-learn>=1.3"
install_pkg "matplotlib"   "matplotlib>=3.7"

# ── Medical imaging ───────────────────────────────────────────────────────────
install_pkg "nibabel"      "nibabel>=5.0"
install_pkg "pydicom"      "pydicom>=2.4"

# ── SimpleITK (registration) — try multiple sources ──────────────────────────
echo -e "  ${CY}→${R} Installing SimpleITK (registration)..."
if pip install SimpleITK --quiet 2>/dev/null; then
    ok "SimpleITK"
elif pip install SimpleITK-SimpleElastix --quiet 2>/dev/null; then
    ok "SimpleITK (via SimpleElastix)"
else
    warn "SimpleITK failed — CT registration will use basic resampling"
    warn "Try later:  pip install SimpleITK"
fi

# ── Visualisation ─────────────────────────────────────────────────────────────
install_pkg "plotly"       "plotly>=5.18"
install_pkg "Pillow"       "pillow>=10.0"

# ── PDF report ────────────────────────────────────────────────────────────────
echo -e "  ${CY}→${R} Installing reportlab (PDF reports)..."
if pip install reportlab --quiet 2>/dev/null; then
    ok "reportlab"
elif pip install "reportlab==3.6.13" --quiet 2>/dev/null; then
    ok "reportlab 3.6.13 (stable fallback)"
else
    warn "reportlab failed — PDF reports will be skipped"
    warn "Try later:  pip install reportlab"
fi

# ── Utilities ─────────────────────────────────────────────────────────────────
install_pkg "tqdm"         "tqdm>=4.65"
install_pkg "colorlog"     "colorlog>=6.7"

echo ""
ok "Package installation complete"

# =============================================================================
# STEP 7 — VERIFY
# =============================================================================
step 7 "Verifying installation"
echo ""

python3 - << 'PYVERIFY'
import sys

packages = [
    ("torch",        "PyTorch",       True),
    ("monai",        "MONAI",         True),
    ("nibabel",      "nibabel",       True),
    ("numpy",        "numpy",         True),
    ("scipy",        "scipy",         True),
    ("skimage",      "scikit-image",  True),
    ("sklearn",      "scikit-learn",  True),
    ("matplotlib",   "matplotlib",    True),
    ("plotly",       "plotly",        False),
    ("SimpleITK",    "SimpleITK",     False),
    ("pydicom",      "pydicom",       False),
    ("reportlab",    "reportlab",     False),
    ("tqdm",         "tqdm",          False),
]

ok_count   = 0
fail_req   = 0
fail_opt   = 0

for mod, name, required in packages:
    try:
        m   = __import__(mod)
        ver = getattr(m, "__version__", "installed")
        print(f"  \033[92m✅\033[0m  {name:<20} {ver}")
        ok_count += 1
    except ImportError:
        tag = "\033[91m❌  REQUIRED\033[0m" if required else "\033[93m⚠️   optional\033[0m"
        print(f"  {tag}  {name}")
        if required:
            fail_req += 1
        else:
            fail_opt += 1

print()
print(f"  Installed : {ok_count}/{len(packages)}")

# Check MPS (Apple Silicon GPU)
try:
    import torch
    mps = torch.backends.mps.is_available()
    cpu = True
    print(f"  MPS (Apple GPU) : {'✅ available' if mps else '⚠️  not available'}")
    print(f"  CPU             : ✅ available")
    print(f"  Model device    : CPU (ConvTranspose3D not on MPS yet)")
except Exception:
    pass

print()
if fail_req == 0:
    print("  \033[92m✅  All required packages OK — ready to run!\033[0m")
else:
    print(f"  \033[91m❌  {fail_req} required package(s) missing\033[0m")
    print("  Re-run setup_project.sh to retry")
    sys.exit(1)
PYVERIFY

VERIFY_EXIT=$?

# =============================================================================
# DONE
# =============================================================================
echo ""
echo -e "${CY}${BOLD}══════════════════════════════════════════════════${R}"
if [ $VERIFY_EXIT -eq 0 ]; then
    echo -e "${GR}${BOLD}  ✅  SETUP COMPLETE!${R}"
else
    echo -e "${YL}${BOLD}  ⚠️   SETUP COMPLETE WITH WARNINGS${R}"
fi
echo -e "${CY}${BOLD}══════════════════════════════════════════════════${R}"

echo ""
echo -e "${BOLD}  PYCHARM SETUP (do this once):${R}"
echo "  1. File → Open → $PROJECT_DIR"
echo "  2. Settings → Python Interpreter → Add Interpreter"
echo "     → Existing environment"
echo "     → Path: $VENV_DIR/bin/python3"
echo "  3. Settings → Run/Debug → Working directory:"
echo "     $PROJECT_DIR"
echo ""
echo -e "${BOLD}  HOW TO RUN THE PIPELINE:${R}"
echo ""
echo "  Option A — Interactive wizard (recommended):"
echo "    source $VENV_DIR/bin/activate"
echo "    cd $PROJECT_DIR"
echo "    python3 run_pipeline.py"
echo ""
echo "  Option B — Terminal shortcut (add to ~/.zshrc):"
echo "    alias pybrain='source $VENV_DIR/bin/activate && cd $PROJECT_DIR && python3 run_pipeline.py'"
echo "    Then just type:  pybrain"
echo ""
echo "  Setup log saved → $LOG_FILE"
echo ""

# Add/Update alias to .zshrc
ALIAS_STR="alias pybrain='source $VENV_DIR/bin/activate && cd $PROJECT_DIR && python3 run_pipeline.py'"
ZSH_PATH="$HOME/.zshrc"

if grep -q "alias pybrain" "$ZSH_PATH" 2>/dev/null; then
    # Use perl for safe in-place replacement with variables
    perl -i -pe "s|^alias pybrain=.*|$ALIAS_STR|" "$ZSH_PATH"
    ok "Updated 'pybrain' shortcut in ~/.zshrc to point here"
else
    echo "" >> "$ZSH_PATH"
    echo "# PY-BRAIN pipeline launcher" >> "$ZSH_PATH"
    echo "$ALIAS_STR" >> "$ZSH_PATH"
    ok "Added 'pybrain' shortcut to ~/.zshrc"
fi
info "Open a new Terminal and type: pybrain"
