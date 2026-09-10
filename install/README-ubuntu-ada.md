# Installation Ubuntu sur cartes Ada (sm_89), sans compilation

Cible : Dell R750xa, RTX 4000 Ada 20 Go (sm_89), Ubuntu, pilote 595 (CUDA 13.2).
Vaut pour toute RTX 40 / RTX Ada.

```bash
./install/install-ubuntu-ada.sh            # ou --models pour pre-telecharger
~/ai/run-comfy.sh
```

## Pourquoi aucune compilation n'est necessaire

Le message `CUDA error: no kernel image is available` sur Ada ne vient pas d'un
defaut de l'installation mais du contenu des roues. Chaque bibliotheque native
CUDA embarque un *fatbin* : un conteneur qui regroupe plusieurs jeux de code
machine, un par architecture. Un jeu compile pour `sm_86` tourne sur `sm_89`
(compatibilite binaire garantie a l'interieur d'une meme generation majeure),
mais un jeu `sm_120` ne tourne pas sur `sm_89`, et son PTX ne peut pas etre
retro-traduit vers une architecture anterieure.

Contenu reel des trois jeux de roues du depot, mesure avec
`install/check_wheel_arch.py` (parseur d'en-tetes fatbin) :

| Jeu de roues | cumesh | flex_gemm / nvdiffrast / nvdiffrec_render / o_voxel | Ada sm_89 |
|---|---|---|---|
| `Torch2110` (cp313) | sm_80, 86, 100, 120 | sm_80, 86, 100, 120 | **oui** |
| `Torch291` (cp312) | sm_86 | **sm_120 seul** | non |
| `Torch270` (cp312) | sm_86 | sm_86 (o_voxel : 80, 86, 89, 90, 120) | oui |

Le jeu `Torch2110` couvre donc Ada d'origine. Recompiler `flex_gemm`, `o_voxel`
et `nvdiffrast` depuis les sources — avec CUDA 12.8, gcc-14 et le correctif de
`crt/math_functions.h` — n'a d'objet que si l'on reste sur le jeu `Torch291`.

Verification a tout moment :

```bash
python install/check_wheel_arch.py --installed --require 8.9
```

L'etape 11 du script va plus loin : elle rasterise reellement un triangle avec
nvdiffrast. C'est le seul test qui execute du code machine sur la carte, donc
le seul qui distingue une installation complete d'une installation qui
n'echouera qu'au premier rendu.

## Le triplet de versions

| Element | Version | Raison |
|---|---|---|
| Python | 3.13 | les roues `Torch2110` sont `cp313` |
| torch | 2.11.0+cu130 | les roues sont liees a `libcudart.so.13` ; un build cu128 echoue a l'import |
| torchvision | 0.26.0 | seule version declarant `torch==2.11.0` |

torch embarque sa propre `libcudart` : le toolkit CUDA 13.3 du systeme n'est pas
sollicite. Le pilote 595 (CUDA 13.2) suffit.

Les remontees silencieuses de torch cassent l'ABI des roues. Le script ecrit
`$VENV/pip.conf` avec un fichier de contraintes : tout `pip install` ulterieur
dans ce venv, y compris lance par un autre installateur, reste verrouille.

## Points de vigilance

- **`open3d` n'a pas de roue cp313.** Il est exclu des `requirements.txt` par le
  script. Il n'est importe qu'a l'interieur d'une fonction (`nodes.py`, noeud de
  nettoyage de maillage) : rien d'autre ne casse, `pymeshlab` et `meshlib`
  couvrent les memes operations.
- **`plyfile`, `easydict`, `trimesh`, `zstandard` sont a installer a la main.**
  Les roues natives sont posees avec `--no-deps`, faute de quoi pip ferait
  remonter torch et casserait l'ABI. Leurs dependances Python ne sont alors
  declarees nulle part : `o_voxel` importe `plyfile` des le chargement du
  module, et le fork GGUF a besoin d'`easydict`. Sans elles, les deux noeuds
  Trellis2 echouent a l'import au demarrage de ComfyUI.
- **Ne pas lancer `install.py` du fork GGUF tel quel.** Il retelecharge ses
  propres roues CUDA depuis ses releases, sans garantie d'architecture, et
  ecraserait le jeu `Torch2110`. Le script n'appelle que sa fonction
  `apply_patches()`, qui remplace `remeshing.py` dans cumesh et
  `flexible_dual_grid.py` dans o_voxel.
- **Texture en Q4_K_M : couleurs delirantes.** Le modele de flux de texture ne
  survit pas a cette quantification (la couleur est un signal haute frequence,
  bien plus sensible que la geometrie). Utiliser Q8_0 pour la texture, ou le
  modele officiel BF16. Sur 20 Go la question ne se pose pas.
- **`use_tiled_encoder` / `use_tiled_decoder_for_texture`** (noeud MeshTexturing
  du fork GGUF) appellent `SparseUnetVaeDecoder._tiled_forward`, methode qui
  n'existe pas dans ce depot (verifie). Les laisser a `false`.
- **Une seule carte.** Le pic memoire vient du remaillage
  (`cumesh.remeshing.remesh_narrow_band_dc`), sur un maillage contigu de
  plusieurs millions de triangles : ni lot ni couches, rien a repartir entre
  GPU. `run-comfy.sh` fixe donc `CUDA_VISIBLE_DEVICES=0`.

## Chemins des modeles

Attendus par `nodes.py` sous `ComfyUI/models/` :

- `facebook/dinov3-vitl16-pretrain-lvd1689m/` — telecharge automatiquement
  depuis le miroir `visualbruno/dinov3-vitl16-pretrain-lvd1689m` (pas d'acces
  restreint a demander).
- `microsoft/TRELLIS.2-4B/`

## Sources

- visualbruno/ComfyUI-Trellis2, signalement 182 : recette Fedora (Liatach),
  cause racine sm_89 et configuration Ubuntu validee sur RTX 4070 Ti SUPER
  (n00n0i), absence de roue open3d cp313 (Lozano71).
- Les architectures du tableau ci-dessus sont mesurees sur les roues de ce
  depot, pas reprises d'un tiers : `install/check_wheel_arch.py` les relit.
