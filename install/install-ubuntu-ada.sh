#!/usr/bin/env bash
#
# Installation de ComfyUI + TRELLIS.2 sur Ubuntu avec des cartes Ada (sm_89,
# RTX 40 / RTX 4000-6000 Ada). Aucune compilation : le jeu de roues
# wheels/Linux/Torch2110 embarque des cubins sm_80 et sm_86, binairement
# compatibles avec sm_89. Voir install/README-ubuntu-ada.md.
#
# Usage :
#   ./install/install-ubuntu-ada.sh                 # installation complete
#   ./install/install-ubuntu-ada.sh --models        # + pre-telechargement des modeles
#   TRELLIS_ROOT=/data/ai ./install/install-ubuntu-ada.sh
#
set -euo pipefail

# ---------------------------------------------------------------- parametres
TRELLIS_ROOT="${TRELLIS_ROOT:-$HOME/ai}"
VENV="${VENV:-$TRELLIS_ROOT/comfy-env}"
COMFY="${COMFY:-$TRELLIS_ROOT/ComfyUI}"
PY_VERSION="3.13"
TORCH_VERSION="2.11.0"
TORCHVISION_VERSION="0.26.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"
WHEEL_SET="Torch2110"
GGUF_NODE_URL="https://github.com/Aero-Ex/ComfyUI-Trellis2-GGUF"
CITY96_NODE_URL="https://github.com/city96/ComfyUI-GGUF"
COMFY_URL="https://github.com/comfyanonymous/ComfyUI"
SMART_UV_WHL="https://github.com/Aero-Ex/Smart-UV-Projection/releases/download/v0.1.0/smart_uv_projection-0.1.0-py3-none-any.whl"

TRELLIS_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WANT_MODELS=0
[[ "${1:-}" == "--models" ]] && WANT_MODELS=1

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
die()  { printf '\n\033[31mERREUR : %s\033[0m\n' "$*" >&2; exit 1; }

clone_or_update() {
    local url="$1" dest="$2"
    if [[ -d "$dest/.git" ]]; then
        info "maj $(basename "$dest")"
        git -C "$dest" pull --ff-only || info "pull impossible (modifications locales) : on garde l'existant"
    else
        git clone "$url" "$dest"
    fi
}

# --------------------------------------------------------- 1. controles GPU
step "1. Controle du materiel"
command -v nvidia-smi >/dev/null || die "nvidia-smi introuvable : pilote NVIDIA absent."
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | sed 's/^/   /'
CAPS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | tr '\n' ' ')"
if [[ "$CAPS" != *"8.9"* ]]; then
    info "ATTENTION : aucune carte sm_89 detectee (vu : $CAPS)."
    info "Le jeu de roues $WHEEL_SET fournit sm_80, sm_86, sm_100 et sm_120."
fi

# ----------------------------------------------------------- 2. Python 3.13
step "2. Interpreteur Python $PY_VERSION"
PYBIN=""
if command -v "python$PY_VERSION" >/dev/null; then
    PYBIN="$(command -v python$PY_VERSION)"
elif command -v uv >/dev/null; then
    info "python$PY_VERSION absent du systeme, installation via uv"
    uv python install "$PY_VERSION"
    PYBIN="$(uv python find "$PY_VERSION")"
else
    die "Ni python$PY_VERSION ni uv. Installer l'un des deux :
      sudo apt install python$PY_VERSION python$PY_VERSION-venv
      ou : curl -LsSf https://astral.sh/uv/install.sh | sh"
fi
info "$PYBIN ($("$PYBIN" -V))"

# ------------------------------------------------------------- 3. venv + pip
step "3. Environnement virtuel : $VENV"
mkdir -p "$TRELLIS_ROOT"
[[ -d "$VENV" ]] || "$PYBIN" -m venv "$VENV"
# shellcheck source=/dev/null
source "$VENV/bin/activate"
python -V | grep -q "$PY_VERSION" || die "le venv n'est pas en Python $PY_VERSION (le supprimer et relancer)"

# Les roues precompilees sont cp313 et liees a une ABI PyTorch precise. Toute
# remontee silencieuse de torch casse l'ensemble : on la verrouille au niveau
# de pip pour tout ce qui sera installe ensuite, y compris par d'autres scripts.
cat > "$VENV/constraints.txt" <<EOF
torch==$TORCH_VERSION
torchvision==$TORCHVISION_VERSION
EOF
cat > "$VENV/pip.conf" <<EOF
[global]
constraint = $VENV/constraints.txt
EOF
python -m pip install --upgrade pip wheel setuptools >/dev/null
info "torch verrouille a $TORCH_VERSION par $VENV/pip.conf"

# ---------------------------------------------------------------- 4. PyTorch
step "4. PyTorch $TORCH_VERSION (build cu130)"
# Les roues Torch2110 sont liees a libcudart.so.13 : un build cu128 echoue a
# l'import. Le paquet torch embarque sa propre libcudart, le toolkit systeme
# (13.3 ici) n'intervient pas.
python -m pip install --index-url "$TORCH_INDEX" \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
python - <<'EOF'
import torch
print("   torch %s / cuda %s / dispo: %s" % (torch.__version__, torch.version.cuda, torch.cuda.is_available()))
if torch.cuda.is_available():
    print("   carte 0 : %s (sm_%d%d)" % (torch.cuda.get_device_name(0), *torch.cuda.get_device_capability(0)))
EOF

# ----------------------------------------------------------------- 5. ComfyUI
step "5. ComfyUI"
clone_or_update "$COMFY_URL" "$COMFY"
python -m pip install -r "$COMFY/requirements.txt"

# ------------------------------------------------------------ 6. custom nodes
step "6. Noeuds personnalises"
mkdir -p "$COMFY/custom_nodes"
NODE_LINK="$COMFY/custom_nodes/ComfyUI-Trellis2"
if [[ "$(dirname "$TRELLIS_SRC")" == "$COMFY/custom_nodes" ]]; then
    info "ce depot est deja dans custom_nodes"
elif [[ -e "$NODE_LINK" && ! -L "$NODE_LINK" ]]; then
    info "ATTENTION : $NODE_LINK existe deja et n'est pas un lien : laisse en l'etat"
else
    ln -sfn "$TRELLIS_SRC" "$NODE_LINK"
    info "lien $NODE_LINK -> $TRELLIS_SRC (le developpement se fait dans ce depot)"
fi
clone_or_update "$GGUF_NODE_URL"    "$COMFY/custom_nodes/ComfyUI-Trellis2-GGUF"
clone_or_update "$CITY96_NODE_URL"  "$COMFY/custom_nodes/ComfyUI-GGUF"

# open3d n'a pas de roue cp313. Il n'est importe qu'a l'interieur d'une
# fonction (nodes.py, noeud de nettoyage de maillage) : son absence ne bloque
# rien, pymeshlab et meshlib couvrent les memes operations.
step "7. Dependances Python des noeuds (sans open3d)"
for req in "$TRELLIS_SRC/requirements.txt" \
           "$COMFY/custom_nodes/ComfyUI-Trellis2-GGUF/requirements.txt" \
           "$COMFY/custom_nodes/ComfyUI-GGUF/requirements.txt"; do
    [[ -f "$req" ]] || continue
    filtered="$(mktemp)"
    grep -v '^[[:space:]]*open3d' "$req" > "$filtered" || true
    printf '\n' >> "$filtered"   # certains requirements n'ont pas de saut de ligne final
    info "$(basename "$(dirname "$req")")/requirements.txt"
    python -m pip install -r "$filtered"
    rm -f "$filtered"
done
python -m pip install "$SMART_UV_WHL" || info "Smart-UV-Projection indisponible (methode d'UV 'Smart' desactivee)"

# Les roues natives sont posees avec --no-deps (sinon pip ferait remonter
# torch) : leurs dependances Python ne sont donc declarees nulle part. o_voxel
# importe plyfile des le chargement du module, et le fork GGUF a besoin
# d'easydict pour son monkeypatch. Sans elles, les deux noeuds echouent a
# l'import au demarrage de ComfyUI.
info "dependances des roues natives"
python -m pip install plyfile easydict trimesh zstandard

# ------------------------------------------------- 8. extensions CUDA natives
step "8. Extensions CUDA (jeu $WHEEL_SET, sans compilation)"
python "$TRELLIS_SRC/install/check_wheel_arch.py" --require 8.9 \
    "$TRELLIS_SRC/wheels/Linux/$WHEEL_SET/"*.whl \
    || die "les roues $WHEEL_SET ne couvrent pas sm_89 : verifier le depot"
# --no-deps : ces roues declarent des dependances qui feraient remonter torch.
python -m pip install --force-reinstall --no-deps "$TRELLIS_SRC/wheels/Linux/$WHEEL_SET/"*.whl

# Le fork GGUF remplace deux fichiers Python dans cumesh et o_voxel (remaillage
# et maillage par tuiles). On appelle sa fonction de patch sans lancer son
# installateur complet : celui-ci retelechargerait ses propres roues CUDA, dont
# l'architecture n'est pas garantie sm_89.
step "9. Patchs du fork GGUF (remeshing.py, flexible_dual_grid.py)"
GGUF_DIR="$COMFY/custom_nodes/ComfyUI-Trellis2-GGUF"
if [[ -f "$GGUF_DIR/install.py" ]]; then
    python - "$GGUF_DIR" <<'EOF'
import importlib.util, sys
gguf_dir = sys.argv[1]
spec = importlib.util.spec_from_file_location("trellis2_gguf_install", gguf_dir + "/install.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.apply_patches()
EOF
else
    info "install.py absent : patchs non appliques"
fi

# ------------------------------------------------------------- 10. modeles
if [[ "$WANT_MODELS" -eq 1 ]]; then
    step "10. Modeles (pre-telechargement)"
    python -m pip install --upgrade "huggingface_hub[cli]"
    # Les noeuds telechargent d'eux-memes au premier lancement ; on prend de
    # l'avance. DINOv3 passe par le miroir non restreint utilise par le depot.
    hf download visualbruno/dinov3-vitl16-pretrain-lvd1689m \
        --local-dir "$COMFY/models/facebook/dinov3-vitl16-pretrain-lvd1689m"
    hf download microsoft/TRELLIS.2-4B \
        --local-dir "$COMFY/models/microsoft/TRELLIS.2-4B"
else
    info "(modeles non pre-telecharges : relancer avec --models, sinon les noeuds les recuperent au premier rendu)"
fi

# ------------------------------------------------------- 11. test de charge
step "11. Verification : lancement reel d'un noyau CUDA"
python - <<'EOF'
import sys
import torch
print("   torch %s, carte %s, sm_%d%d" % (
    torch.__version__, torch.cuda.get_device_name(0), *torch.cuda.get_device_capability(0)))

erreurs = []
for module in ("cumesh", "flex_gemm", "o_voxel", "nvdiffrec_render"):
    try:
        __import__(module)
        print("   import %-18s OK" % module)
    except Exception as exc:
        erreurs.append("%s : %s" % (module, exc))
        print("   import %-18s ECHEC : %s" % (module, exc))

# nvdiffrast : on rasterise un triangle. C'est le test qui compte, il execute
# vraiment du code machine sur la carte ; un jeu de roues sans cubin compatible
# echoue ici avec "no kernel image is available".
try:
    import nvdiffrast.torch as dr
    ctx = dr.RasterizeCudaContext()
    pos = torch.tensor([[[-0.8, -0.8, 0.0, 1.0],
                         [ 0.8, -0.8, 0.0, 1.0],
                         [ 0.0,  0.8, 0.0, 1.0]]], device="cuda")
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    rast, _ = dr.rasterize(ctx, pos, tri, resolution=[64, 64])
    couverture = float((rast[..., 3] > 0).float().mean())
    print("   noyau nvdiffrast   OK (%.0f%% du cadre couvert par le triangle)" % (100 * couverture))
except Exception as exc:
    erreurs.append("nvdiffrast : %s" % exc)
    print("   noyau nvdiffrast   ECHEC : %s" % exc)

if erreurs:
    print("\n   Installation incomplete :")
    for e in erreurs:
        print("     - %s" % e)
    sys.exit(1)
EOF

# ------------------------------------------------------ 12. script de lancement
step "12. Script de lancement"
cat > "$TRELLIS_ROOT/run-comfy.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
# Une seule carte : le pic memoire vient du remaillage, sur un objet contigu
# qui ne se decoupe pas entre GPU. Ajuster l'index selon la carte libre.
export CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-0}"
# Le remaillage alloue et libere de gros blocs : les segments extensibles
# evitent la fragmentation qui provoque des OOM a memoire pourtant disponible.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$VENV/bin/activate"
cd "$COMFY"
# 127.0.0.1 n'ouvre que la boucle locale. COMFY_LISTEN=0.0.0.0 expose
# l'interface a tout le reseau local : ComfyUI n'a aucune authentification.
exec python main.py --listen "\${COMFY_LISTEN:-127.0.0.1}" --port "\${COMFY_PORT:-8188}" "\$@"
EOF
chmod +x "$TRELLIS_ROOT/run-comfy.sh"

step "Termine"
info "lancer : $TRELLIS_ROOT/run-comfy.sh"
info "verifier les architectures installees : python $TRELLIS_SRC/install/check_wheel_arch.py --installed --require 8.9"
