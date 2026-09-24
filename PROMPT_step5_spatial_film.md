# Prompt: guide me through implementing Spatial FiLM (SpFiLM) in `code/spfilm`

## 0. Your role and how this session works

You are my implementation guide for **Step 5 of a research project: adding Spatial
FiLM (SpFiLM) as a third experimental arm** to an existing, tested PyTorch codebase
for optic disc / cup segmentation in retinal fundus images. I write the code. You
do not.

Work **one piece at a time**, in the order of Section 5. For each piece:

1. Explain in a few sentences what the piece is for and how it plugs into what
   already exists (Section 3 tells you exactly what exists).
2. Give me the **contract** to implement: file path, class/function signatures,
   docstring intent, invariants, and the tests that must pass. Contracts, not
   implementations. If I ask for a hint, give the smallest hint that unblocks me.
3. Stop and wait for me to implement it and run the tests.
4. When I paste my code back, review it against the contract and the reference
   implementation (Section 2). Point at concrete deviations; do not rewrite it.
5. Only then move to the next piece.

You do not have the repository. Section 3 quotes the signatures you need
verbatim. If you need to see a file, ask me to paste it. When something in the
repo contradicts this prompt, the repo wins — say so and adapt.

Code style: match the surrounding code — explanatory comments that say *why*,
type hints, `from __future__ import annotations`, `ValueError` for contract
violations in layers, `unittest` (no pytest), tests named as sentences
(`test_gamma_and_beta_zero_is_the_identity`). Never touch the existing
`GlobalFiLM` / `ConditionedUNet` code paths except where a contract says so;
every experimental arm must differ from its comparator in exactly one thing.

## 1. Context

**Task.** 2-channel (disc, cup) segmentation of 2D fundus images, 512×512
letterboxed RGB in [0, 1]. Four acquisition domains: `refuge_zeiss`,
`refuge_canon_val`, `drishti_gs`, `rim_one_dl`. Step 4 dropped RIM-ONE-DL; the
**active set is the three others**. RIM-ONE stays configured in every config
(`domains` block) because the locked split manifests cover all four and are
revalidated from that block — only `protocol.*_domains` decides who takes part.

**Backbone.** A 5-level 2D U-Net (`PlainUNet`, widths `base×(1,2,4,8,16)`,
`base_channels=16`, InstanceNorm2d(affine) + ReLU, `DoubleConv` blocks). Trained
with BCE+Dice, Adam 1e-3, cosine LR, 300 epochs, early stopping in "monitor"
mode (never terminates), fp16 autocast on CUDA. Augmentation: h-flip, ±10°
rotation, ±10% brightness/contrast. No scale augmentation.

**What is already done (Step 4, Global FiLM).** Channel-wise FiLM after every
encoder `DoubleConv` (before pooling) and after the bottleneck; decoder
unconditioned; frozen one-hot domain embedding (64-d); MLP 64→256→256→2C;
`(1+γ)F+β`; clamp ±5; fp32 under autocast. It matches the supervisor's reference
implementation block for block. Three protocols exist, all over the 3 active
domains with fixed budgets (40 train / 10 val / 50 test images per domain):

| regime | folds | test-time code | runner | status |
|---|---|---|---|---|
| train-on-all | train on 3 (120/30), test each domain (50) | oracle (own code) | `run_stage4_all_domains.py` | **done, 5+5 seeds**: FiLM ≤ plain (Δ ≈ −0.006/−0.014 Drishti disc/cup; 3/6 cells significantly worse after Holm, 0/6 better). Own code best in 30/30 seed-cells; wrong-code penalty 0.05–0.10 disc, 0.11–0.26 cup, so the FiLM layers *are* used. |
| leave-one-domain-out | train on 2 (80/20), test held-out (50) | one code per held-out domain: nearest source centroid (FOV RGB mean/std) to the held-out domain's unlabelled reference sample | `run_stage3_lodo_3_1_fixed.py` | built and smoked; 30 jobs not launched |
| train-on-one | train on 1 (40/10), test other 2 | degenerate (one code) | `run_stage3_lodo_1_3.py` | control only, shelved |

**Why SpFiLM now.** Global FiLM did not beat the plain U-Net even when the domain
is known. The domain shift here is *geometric*, not photometric: the disc
diameter / frame ratio is 1.25× larger in Drishti than in REFUGE, and intensity
distances anti-correlate with Dice. A per-channel affine cannot encode *where* in
the frame to modulate; SpFiLM can. The experimental question for Step 5 is
whether spatially-varying modulation recovers what channel-wise modulation
could not, first in the train-on-all regime (where the comparison is cleanest),
then in LODO.

**Reference implementation (the "how").** The supervisor's own SpFiLM code, for
3D brain parcellation across two MRI contrasts:
<https://github.com/p-singh-kcl/spatial_film_parcellation>. Read, in this order:

- `models/spatial_film.py` — `SpatialFiLM3d` (the layer) and
  `UNetSpatialFiLMOneHot3D` (`forward_encoder` shows where it is inserted and
  what image tensor it is fed at each level). This is the file to mirror.
- `models/film_mlp.py` — his Global FiLM; ours already matches it.
- `models/embeddings.py` — `OneHotContrastEmbedding` (ours: `DomainOneHot`).
- `models/blocks.py` — `ConvBlock3d`, `Decoder3D`, `DEFAULT_CHANNELS`.
- `configs/spatial_film_onehot_k{1,2,4,8,16}.yaml` — the rank sweep he ran.

Section 2 states the method as the paper defines it, and the 3D→2D mapping.

## 2. The method, and how it maps to this codebase

From the draft (Sec. 2.1–2.2). For a feature map `F ∈ R^{C×H'×W'}` at one encoder
stage and domain code `s`:

- Channel-wise FiLM (what exists): `FiLM(F_c) = (1 + γ_c(s)) F_c + β_c(s)` (Eq. 1).
- SpFiLM lets γ and β vary over pixels `v`, with a **rank-K factorisation**:

  ```
  γ_{c,v}(X, s) = γ̄_c(s) + Σ_{k=1..K} A_{ck}(s) · φ_{k,v}(X)          (Eq. 2)
  β_{c,v}(X, s) = β̄_c(s) + Σ_{k=1..K} B_{ck}(s) · ψ_{k,v}(X)
  SpFiLM(F_{c,v}) = (1 + m_v γ_{c,v}) F_{c,v} + m_v β_{c,v}              (Eq. 3, m = mask gating)
  ```

  With **K = 0 the sums vanish and Eq. 2 collapses to Eq. 1** — this equivalence
  is a hard requirement and a unit test, not a hope.

- **Image-conditioned bases** `φ(X'), ψ(X') ∈ R^{K×H'×W'}`: two *independent*
  generators per conditioning layer (one for scale, one for shift), **shared
  across domains** (they see the image, not the code). The input image is first
  resampled to the feature resolution (`X'`), then each generator is:
  `Conv(in→16, 3×3, stride 2, pad 1, no bias) → InstanceNorm(16, affine) →
  LeakyReLU(0.01) → Conv(16→K, 3×3, pad 1, no bias) → InstanceNorm(K, affine) →
  tanh`, then interpolated back up to `H'×W'`. The stride-2 conv means the bases
  are learned one level coarser than `F` — a deliberate smoothness regulariser.
- **Code-conditioned coefficients**: one MLP `64 → 256 → 256 → 2C(1+K)` maps the
  one-hot code to `γ̄ ∈ R^C`, `A ∈ R^{C×K}`, `β̄ ∈ R^C`, `B ∈ R^{C×K}`.
- γ and β clamped to [−5, 5] **after** assembly; whole modulation in fp32 under
  autocast; result cast back to the feature dtype.
- Insertion: after every encoder conv block, before pooling; bottleneck
  included; decoder unconditioned. Identical to where Global FiLM sits here.

Reference `forward` (from `models/spatial_film.py`, 3D), so you can check my 2D
version against it:

```python
phi = self.spatial_basis_gamma(iv); phi = F.interpolate(phi, size=(D,H,W), mode="trilinear", align_corners=False)
psi = self.spatial_basis_beta(iv);  psi = F.interpolate(psi, size=(D,H,W), mode="trilinear", align_corners=False)
cond = self.conditioning_mlp(contrast_embed.float())
gamma_bar = cond[:, :C]; A = cond[:, C:C+C*K].view(Bs, C, K)
beta_bar  = cond[:, C+C*K:2*C+C*K]; B_mat = cond[:, 2*C+C*K:].view(Bs, C, K)
gamma = gamma_bar.view(Bs,C,1,1,1) + torch.bmm(A, phi.view(Bs,K,-1)).view(Bs,C,D,H,W)
beta  = beta_bar.view(Bs,C,1,1,1)  + torch.bmm(B_mat, psi.view(Bs,K,-1)).view(Bs,C,D,H,W)
gamma = gamma.clamp(-5, 5); beta = beta.clamp(-5, 5)
if brain_mask is not None: m = F.interpolate(brain_mask.float(), size=(D,H,W), mode="nearest"); gamma *= m; beta *= m
return ((1 + gamma) * features_f + beta).to(features.dtype)
```

In his encoder, level 0 gets the raw input, later levels get
`F.interpolate(x, size=feature_hw, mode=…, align_corners=False)`.

**3D → this codebase mapping (fixed decisions):**

| reference (brain MRI, 3D) | here (fundus, 2D) |
|---|---|
| `Conv3d`, `InstanceNorm3d`, trilinear | `Conv2d`, `InstanceNorm2d`, bilinear (`align_corners=False`) |
| 1-channel z-normalised volume | 3-channel letterboxed RGB in [0, 1]; basis stack's first conv takes `in_channels=3` (make it a constructor arg so a 1-channel luminance variant is one line) |
| contrast code `s ∈ {T1, T1c}` | fold-local domain index into `DomainVocabulary` (sorted domain names that saw gradients); frozen one-hot `DomainOneHot(64)` |
| `film_levels=[0,1,2,3,4]` | `film_levels: int` counting from the shallowest; 5 = all four encoder blocks + bottleneck |
| brain-mask gating | **FOV gating**: optional, default **off**; mask = pixels whose Rec.601 luminance exceeds `FOV_LUMINANCE_THRESHOLD` (`spfilm.global_histograms`), computed once from the input, nearest-resampled per level. The letterbox border is black and only Drishti is non-square, so this mostly matters there. |
| `rank` K=8 default, swept {1,2,4,8,16} | `film.rank`, required explicitly for the spatial arm; first run K=8 |
| deep supervision, AdamW, 128³ patches | unchanged from Step 4 here (BCE+Dice, Adam, whole 512² image); do not import his training details |

## 3. What exists — the seams SpFiLM plugs into

All paths relative to `code/spfilm/`. Python: `.spfilm/bin/python`. Tests:
`.spfilm/bin/python -m unittest discover tests` (449 pass today).

### 3.1 The layer to mirror — `src/spfilm/film/global_film.py`

```python
DEFAULT_EMBEDDING_DIM = 64; DEFAULT_HIDDEN_DIM = 256; DEFAULT_CLAMP = 5.0

class DomainOneHot(nn.Module):
    def __init__(self, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> None   # buffer eye(embedding_dim), no params
    def forward(self, domain_index: torch.Tensor) -> torch.Tensor           # (N,) long -> (N, embedding_dim)

class GlobalFiLM(nn.Module):
    def __init__(self, num_channels: int, embedding_dim=64, hidden_dim=256, clamp=5.0) -> None
        self.generator = nn.Sequential(Linear(embedding_dim, hidden), ReLU, Linear(hidden, hidden), ReLU, Linear(hidden, 2*num_channels))
    def gamma_beta(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]   # each (N, C), clamped, fp32 under autocast
    def forward(self, features: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor   # (N,C,H,W); (1+γ)F+β in fp32, cast back
```

Its module docstring ends: *"SpFiLM with K=0 must reproduce this layer
numerically; keep them aligned."* Name your MLP `generator` too, so a test can
copy `GlobalFiLM.generator.state_dict()` into the K=0 SpFiLM and assert equality.

### 3.2 The network to mirror — `src/spfilm/model.py`

```python
class ConditionedUNet(nn.Module):
    ENCODER_LEVELS = 5
    def __init__(self, num_domains, in_channels=3, out_channels=2, base_channels=32,
                 film_levels=ENCODER_LEVELS, embedding_dim=64, hidden_dim=256, clamp=5.0) -> None
        self.backbone = PlainUNet(...)            # a real PlainUNet: state-dict keys are the plain keys under "backbone."
        self.one_hot = DomainOneHot(embedding_dim)
        widths = (c1, c1*2, c1*4, c1*8, c1*16)
        self.films = nn.ModuleList(GlobalFiLM(w, ...) for w in widths[:film_levels])
    def forward(self, inputs: torch.Tensor, domain_index: torch.Tensor) -> torch.Tensor:
        embedding = self.one_hot(domain_index)
        skip1 = self._film(0, net.down1.convolutions(inputs), embedding)
        skip2 = self._film(1, net.down2.convolutions(net.down1.pool(skip1)), embedding)
        skip3 = self._film(2, net.down3.convolutions(net.down2.pool(skip2)), embedding)
        skip4 = self._film(3, net.down4.convolutions(net.down3.pool(skip3)), embedding)
        features = self._film(4, net.bottleneck(net.down4.pool(skip4)), embedding)
        ... net.up1..up4, net.output

class SpatialFiLMUNet():      # <-- committed placeholder, empty. Replace it.
    pass

ARMS = ("plain", "global_film")

def build_model(arm, base_channels, num_domains=None, film_levels=5, embedding_dim=64, hidden_dim=256, clamp=5.0) -> nn.Module
    # "plain" -> PlainUNet; "global_film" -> ConditionedUNet (needs num_domains);
    # there is a stray `if arm == "spatial_film": return SpatialFiLMUNet` that returns the CLASS — fix it.
```

`src/spfilm/film/spfilm.py` exists and is **empty** — that is where the layer goes.

### 3.3 The engine — `src/spfilm/engine.py` (nothing here branches on `"global_film"`; everything keys on `arm != "plain"`)

```python
@dataclass
class Stage2Config:
    ...
    arm: str = "plain"
    film_levels: int = 5
    film_embedding_dim: int = 64
    film_hidden_dim: int = 256
    film_clamp: float = 5.0
    test_conditioning: str = "nearest_domain"     # one of SELECTION_POLICIES: nearest_domain | nearest_image | oracle

FILM_CONFIG_FIELDS = ("arm", "film_levels", "film_embedding_dim", "film_hidden_dim", "film_clamp", "test_conditioning")

def _resume_fingerprint(config, split_counts) -> str:
    # plain arm pops every FILM_CONFIG_FIELDS key before hashing, so old plain runs still resume.
    # A conditioned arm hashes every field.  <-- see the gotcha in Section 6 before adding fields.

def _forward(model, images, condition: ConditionResult | None) -> torch.Tensor:
    if condition is None: return model(images)
    return model(images, condition.indices)          # <-- the ONLY call into a conditioned model. Signature is fixed: (inputs, domain_index).

# in run_experiment:
conditioned = config.arm != "plain"
vocabulary = DomainVocabulary.from_domains(record.domain for record in splits["train"])   # fold-local codes
selector = NearestDomainSelector.fit_from_loader(...)                                       # image descriptors only
model = build_model(config.arm, config.base_channels, num_domains=len(vocabulary), film_levels=..., embedding_dim=..., hidden_dim=..., clamp=...)
# _conditioning_report(...) writes report["conditioning"] with a "film": {"levels","embedding_dim","hidden_dim","clamp"} block,
# the selector validation, the per-image conditioning CSVs, and the fixed-code sweep (score the test set once per code).
```

Condition providers (`src/spfilm/film/conditioning.py`) — reused unchanged, they
only produce `ConditionResult.indices: (N,) long`:

```python
@dataclass(frozen=True)
class ConditionResult:
    indices: torch.Tensor; source: str; distances: torch.Tensor | None = None; descriptors: torch.Tensor | None = None
class OracleCondition   # true domain; train/val always; test under train-on-all
class DomainCondition   # one code per held-out domain from DomainDecision; test under LODO ("nearest_domain")
class NearestCondition  # per image (ablation, "nearest_image")
class FixedCondition    # fixed-code sweep
```

Because the model receives the *image* as `inputs` anyway, SpFiLM needs **no
engine change** for its basis maps: the network resamples its own input
internally. The engine changes are config plumbing only.

### 3.4 Config parsing — `src/spfilm/stage3_single_source.py` (one parser serves all three runners)

```python
@dataclass(frozen=True)
class Stage3SingleSourceConfig:
    ...
    arm: str = "plain"
    film_levels: int = 5; film_embedding_dim: int = 64; film_hidden_dim: int = 256; film_clamp: float = 5.0
    test_conditioning: str = "nearest_domain"
    paired_arm: str | None = None
    def training_config(self, source_domain, run_seed, output_dir, requested_device=None) -> Stage2Config   # forwards film_* fields

FILM_BLOCK_DEFAULTS = {"film_levels": 5, "film_embedding_dim": 64, "film_hidden_dim": 256, "film_clamp": 5.0, "test_conditioning": "nearest_domain"}

def _parse_film_block(raw, arm) -> dict:
    # plain arm must not carry a film block; allowed keys today:
    allowed = {"levels", "embedding_dim", "hidden_dim", "clamp", "test_conditioning"}
```

`from_json` validates `arm in ARMS` (imported from `model.py`), so adding the
arm to `ARMS` is what makes configs load.

### 3.5 Runners, configs, submit scripts, aggregators — no code changes expected

- Train-on-all: `run_stage4_all_domains.py --config <cfg> {check|run --seed N [--smoke] [--device cpu] [--out-dir …]}`;
  requires `test_conditioning: "oracle"` for any conditioned arm.
  Config to copy: `configs/stage4_all_domains_global_film_3dom{,_create}.json`
  (`experiment_name: stage4_all_domains_fixed_budget_global_film_3dom`,
  `stage: all_domains_fixed_budget`, `film: {levels 5, embedding_dim 64, hidden_dim 256, clamp 5.0, test_conditioning oracle}`,
  `protocol.active_domains`, `protocol.inactive_domains: [rim_one_dl]`, `protocol.paired_arm`).
  Submit to copy: `submit_stage4_all_domains_film.sh` (job `allf_s4`, out-dir `artifacts/runs/allf_s4_seed_<seed>_<job>`, 3 h).
  Aggregate: `aggregate_stage4_all_domains.py --plain-arm … --film-arm … --expected-seeds … --report-out … --csv-out …`.
- LODO: `run_stage3_lodo_3_1_fixed.py`, configs `configs/stage4_global_film_3dom{,_create}.json`
  (`test_conditioning: nearest_domain`), `submit_stage4_global_film.sh`, `aggregate_stage4_film.py`.
- The `_create.json` twin of every config differs only in the four `data_root` values (CREATE paths).
- The existing test file `tests/test_global_film.py` has `Stage4ConfigTests` that
  assert "the FiLM config differs from the plain config only in the arm-only
  keys" — copy that pattern for the new configs.

## 4. Design decisions already made (do not reopen) and open ones (ask me)

**Fixed:** frozen one-hot for all arms; insertion points; decoder unconditioned;
clamp 5; fp32 under autocast; `embedding_dim` 64; MLP hidden 256; `film_levels`
semantics; bases from the input image (not from features); two independent basis
generators; stride-2 basis grid; bilinear resampling; the K=0 identity with
`GlobalFiLM` as a unit test; SpFiLM is a **third arm** — Global FiLM stays
byte-identical.

**Open — flag each when we reach it, default in bold:**
1. FOV gating: **off** for the first run (so SpFiLM − Global FiLM = the spatial term only); on as an ablation.
2. K: **8** first (the supervisor's default config); then {2, 16} if budget allows.
3. Regime order: **train-on-all first** (Global FiLM ≤ plain there; the question is whether the spatial term changes that), then LODO with the per-domain nearest code.
4. Basis input channels: **RGB (3)**; a luminance (1) variant is a constructor arg.
5. `rank: 0` in a *run* config for the spatial arm: **refused** (that is `global_film`, already run); the equivalence lives in the tests.

## 5. The work, as contracts

### Piece 1 — the layer: `src/spfilm/film/spfilm.py`

Module docstring: what SpFiLM is (Eq. 2–3), the K=0 identity with
`GlobalFiLM`, the 3D→2D mapping, the fp32/clamp policy.

```python
DEFAULT_RANK = 8
DEFAULT_BASIS_HIDDEN_CHANNELS = 16

class SpatialBasis(nn.Module):
    """phi(X') or psi(X'): K smooth basis maps in [-1, 1] from the input resampled to feature resolution."""
    def __init__(self, rank: int, in_channels: int = 3, hidden_channels: int = DEFAULT_BASIS_HIDDEN_CHANNELS) -> None
        # rank >= 1 else ValueError. Stack exactly as Section 2 (no bias on either conv; IN affine on both; LeakyReLU(0.01); tanh).
    def forward(self, image: torch.Tensor) -> torch.Tensor
        # image (N, in_channels, H', W') -> (N, K, H', W'): run the stride-2 stack, then bilinear-interpolate back to (H', W').

class SpatialFiLM(nn.Module):
    """Rank-K spatially varying FiLM for one feature map; owns its basis generators and coefficient MLP."""
    def __init__(self, num_channels: int, rank: int = DEFAULT_RANK, embedding_dim: int = 64, hidden_dim: int = 256,
                 clamp: float = 5.0, in_channels: int = 3, basis_hidden_channels: int = 16, fov_gating: bool = False) -> None
        # rank >= 0; rank == 0 -> self.basis_gamma is None and self.basis_beta is None (no parameters, no state-dict keys).
        # self.generator = Linear(embedding_dim, hidden) ReLU Linear(hidden, hidden) ReLU Linear(hidden, 2*C*(1+K))
    def coefficients(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        # -> (gamma_bar (N,C), A (N,C,K), beta_bar (N,C), B (N,C,K)); output layout is [gamma_bar | A | beta_bar | B]; fp32 under autocast.
    def fields(self, image: torch.Tensor, embedding: torch.Tensor, size: tuple[int, int],
               fov_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]
        # -> (gamma (N,C,H',W'), beta (N,C,H',W')), assembled per Eq. 2 with torch.bmm, clamped to [-clamp, clamp] AFTER assembly,
        #    then multiplied by the nearest-resampled fov_mask (N,1,H',W') if gating is on and a mask is given. K=0 -> broadcast bars.
    def forward(self, features: torch.Tensor, image: torch.Tensor, embedding: torch.Tensor,
                fov_mask: torch.Tensor | None = None) -> torch.Tensor
        # (1 + gamma) * features + beta in fp32 under autocast(enabled=False); cast back to features.dtype.
        # ValueError on: features not (N,C,H,W) with C == num_channels; batch mismatch between features/image/embedding;
        # image spatial size != features spatial size (the caller resamples; this layer does not).
```

Invariants to test (`tests/test_spatial_film.py`, class `SpatialBasisTests`,
`SpatialFiLMTests`):
- basis output is `(N, K, H', W')`, in `[-1, 1]`, different for different images, independent of the code.
- **K=0 equals `GlobalFiLM`**: build both with the same `num_channels`, copy
  `GlobalFiLM.generator.state_dict()` into `SpatialFiLM.generator`, assert
  `torch.allclose(atol=1e-6)` on random features/embeddings.
- zero MLP output ⇒ identity (mirror `test_gamma_and_beta_zero_is_the_identity`).
- for K≥1 with nonzero `A`, γ **varies over pixels** (the contrast with
  `test_scale_and_shift_are_per_channel_and_uniform_over_pixels` in the global tests).
- clamp holds on the assembled field, not only on the bars.
- output dtype preserved and finite under `torch.autocast` with large inputs.
- FOV gating: pixels with mask 0 come out exactly equal to the input features.
- parameter count equals the closed form: MLP `(E·H + H) + (H·H + H) + (H·2C(1+K) + 2C(1+K))` plus, if K≥1,
  `2 × [9·in·16 + 2·16 + 9·16·K + 2·K]`.
- `coefficients` layout: perturb one slice of the last `Linear` bias and check only the intended tensor moves.

### Piece 2 — the network: replace the stub in `src/spfilm/model.py`

```python
class SpatialFiLMUNet(nn.Module):
    """PlainUNet with SpFiLM after each encoder block; the K=0 case is ConditionedUNet."""
    ENCODER_LEVELS = 5
    def __init__(self, num_domains: int, in_channels: int = 3, out_channels: int = 2, base_channels: int = 32,
                 film_levels: int = ENCODER_LEVELS, embedding_dim: int = 64, hidden_dim: int = 256, clamp: float = 5.0,
                 rank: int = DEFAULT_RANK, fov_gating: bool = False) -> None
        # self.backbone = PlainUNet(...); self.one_hot = DomainOneHot(embedding_dim);
        # self.films = nn.ModuleList(SpatialFiLM(width, rank, ..., in_channels=in_channels, fov_gating=fov_gating) for width in widths[:film_levels])
    def forward(self, inputs: torch.Tensor, domain_index: torch.Tensor) -> torch.Tensor
        # same skeleton as ConditionedUNet.forward; level 0 is fed `inputs` itself, deeper levels
        # F.interpolate(inputs, size=features.shape[-2:], mode="bilinear", align_corners=False) — resample the IMAGE, never the features.
        # fov mask (if gating): computed once from `inputs` via the FOV luminance rule, passed down; each layer nearest-resamples it.
        # ValueError if any domain_index >= num_domains (mirror ConditionedUNet).

ARMS = ("plain", "global_film", "spatial_film")

def build_model(arm, base_channels, num_domains=None, film_levels=5, embedding_dim=64, hidden_dim=256, clamp=5.0,
                rank: int = DEFAULT_RANK, fov_gating: bool = False) -> nn.Module
    # "spatial_film" -> SpatialFiLMUNet(...) (an INSTANCE); needs num_domains like global_film. The old `return SpatialFiLMUNet` bug goes away.
```

Put the FOV-mask helper where the luminance rule already lives
(`spfilm.global_histograms.FOV_LUMINANCE_THRESHOLD`, `LUMA_WEIGHTS`) or next to
`fov_descriptor` in `film/conditioning.py`, and call it from the network; do not
re-derive the threshold.

Tests (`SpatialFiLMUNetTests`, mirroring `ConditionedUNetTests`):
- spatial shape preserved, 2 output channels, on 3×32×32 inputs (CPU).
- `backbone.*` state-dict keys equal `PlainUNet`'s keys.
- parameter count = backbone + Σ films.
- `film_levels=2` conditions only two levels.
- with every `generator` zeroed **and** K≥1, output equals `PlainUNet` with the
  same backbone weights exactly (the bases are multiplied by zero coefficients).
- untrained code refused.
- `build_model("spatial_film", 8, num_domains=2, rank=4)` is a `SpatialFiLMUNet`;
  without `num_domains` → `ValueError`. Update
  `test_build_model_maps_arms` in `tests/test_global_film.py` only if it breaks
  (it asserts `"spatial"` is unknown, which stays true).

### Piece 3 — config plumbing (four places, all small)

1. `engine.Stage2Config`: add `film_rank: int = DEFAULT_RANK` and
   `film_fov_gating: bool = False` next to the other `film_*` fields; append both
   to `FILM_CONFIG_FIELDS`; pass `rank=config.film_rank, fov_gating=config.film_fov_gating`
   in the `build_model(...)` call; add `"rank": config.film_rank if config.arm == "spatial_film" else 0`
   and `"fov_gating": …` to the `"film"` block in `_conditioning_report`.
2. **Fingerprint** (see the gotcha in Section 6): add
   `SPATIAL_FILM_CONFIG_FIELDS = ("film_rank", "film_fov_gating")` and pop them
   in `_resume_fingerprint` when `config.arm == "global_film"`, so every existing
   Global FiLM fingerprint stays byte-identical and in-flight/relaunched Global
   FiLM jobs keep resuming. The plain arm already pops all film fields.
3. `stage3_single_source.Stage3SingleSourceConfig`: fields `film_rank`,
   `film_fov_gating`; `training_config` forwards them;
   `_parse_film_block`: `allowed |= {"rank", "fov_gating"}`; for `arm ==
   "global_film"` refuse `rank`/`fov_gating` ("the global arm has no spatial
   term"); for `arm == "spatial_film"` **require** `rank` (positive int; 0 is
   refused with "use global_film"); `fov_gating` optional bool.
4. Nothing in the runners or aggregators.

Tests (`SpatialFiLMConfigTests`): rank parsed and forwarded to `Stage2Config`;
missing rank refused for the spatial arm; rank refused for the global arm; rank 0
refused; plain fingerprint unchanged (existing test); **global_film fingerprint
unchanged by the new fields** (compute it with a config built from a dict that
omits the two fields vs one that includes defaults — must be equal); spatial
fingerprint changes with rank.

### Piece 4 — configs, submit script, smoke (train-on-all first)

- `configs/stage4_all_domains_spatial_film_k8_3dom{,_create}.json`: copy the
  Global FiLM twins; change only `arm: "spatial_film"`, `film.rank: 8`
  (`film.fov_gating: false` stated explicitly), `experiment_name:
  "stage4_all_domains_fixed_budget_spatial_film_k8_3dom"`, `output_dir`,
  `protocol.policy`, `protocol.paired_arm` (the plain arm; the Global FiLM arm
  is the second comparison, name it in a `protocol.secondary_comparison` note).
  Keep `domains` complete (all four) and `test_conditioning: "oracle"`.
- Test: the new config differs from the Global FiLM config only in the arm-only
  keys (copy `Stage4ConfigTests.test_film_configs_differ_from_plain_only_in_the_conditioning`).
- `submit_stage4_all_domains_spfilm.sh`: copy `submit_stage4_all_domains_film.sh`;
  job `allsf_s4`, out-dir prefix `allsf_s4_`, new `_create` config. Keep 3 h
  until the smoke's `epoch_seconds` says otherwise (two extra 3×3 convs per level
  on the resampled image plus a `bmm` per level — expect ≤1.5× Global FiLM).
- Verify locally, in this order:
  1. `.spfilm/bin/python -m unittest discover tests` — everything green, including the untouched Step 4 suites.
  2. `.spfilm/bin/python run_stage4_all_domains.py --config configs/stage4_all_domains_spatial_film_k8_3dom.json check --skip-mask-audit`
  3. `.spfilm/bin/python run_stage4_all_domains.py --config configs/stage4_all_domains_spatial_film_k8_3dom.json run --seed 42 --smoke --device cpu --out-dir artifacts/runs/spfilm_smoke_local`
     — expect `arm=spatial_film`, `codes=[drishti_gs, refuge_canon_val, refuge_zeiss]`, `conditioning.film.rank == 8`,
     a fixed-code sweep with three entries, and `parameter_count` = plain + Σ films. Delete the `_smoke` dir after.
- Then, and only after I have agreed the run plan with my supervisor: CREATE smoke,
  seed 42, `aggregate_stage4_all_domains.py --film-arm stage4_all_domains_fixed_budget_spatial_film_k8_3dom --expected-seeds 42`,
  and the same with `--plain-arm stage4_all_domains_fixed_budget_global_film_3dom` for SpFiLM-vs-Global FiLM
  (if the aggregator refuses a conditioned arm as `--plain-arm`, that is a small contract to add, not a reason to hand-compute).

### Piece 5 — LODO variant (after train-on-all has a seed-42 result)

Same pattern from `configs/stage4_global_film_3dom{,_create}.json`
(`test_conditioning: nearest_domain`, `paired_arm:
stage4_lodo_fixed_budget_plain_unet_3dom`), `submit_stage4_spatial_film.sh` (job
`spf_s4`), run through `run_stage3_lodo_3_1_fixed.py`, aggregate with
`aggregate_stage4_film.py --film-arm stage4_lodo_fixed_budget_spatial_film_k8_3dom`.
The per-domain code decision and the fixed-code sweep are inherited unchanged.
Before reading the Dice difference, read each run's
`report["conditioning"]["used_code_matches_best_fixed_code"]` /
`nearest_domain_matches_best_fixed_code` (written by `engine._conditioning_report`
into `test_metrics.json`) and the sweep spread the aggregator prints: the
train-on-all sweep predicts the nearest-intensity rule will hand Drishti the
Zeiss code and vice versa, which were the *worst* codes there.

## 6. Gotchas (each one has bitten this project or the reference)

- **Resume fingerprint.** `_resume_fingerprint` hashes `asdict(config)`. Adding
  fields to `Stage2Config` changes the hash of every conditioned run unless the
  new fields are popped for arms that do not use them. A preempted Global FiLM
  job relaunched after pulling this code would refuse to resume. Piece 3.2 is
  not optional.
- **Conditioned-arm RNG.** Extra modules change the parameter-init order, so
  SpFiLM seed 42 is not "Global FiLM seed 42 plus a spatial term"; the arms are
  paired on test images, not on trajectories.
- **Autocast.** The reference does the entire modulation, *including the basis
  convs*, inside `autocast(enabled=False)`. Keep that: one half-precision
  overflow in a `bmm` turns the whole field into NaN.
- **Clamp after assembly**, not on the bars and coefficients separately; the
  bilinear upsampling can push a value past the bars' range.
- **No bias** on the basis convs (InstanceNorm follows each one).
- **InstanceNorm on K channels** normalises each basis map over its own H'×W':
  fine for K=1, but the smallest grid is the bottleneck at 128 px smoke (8×8 →
  stride 2 → 4×4). Nothing smaller is ever run.
- **The letterbox border** is black and inside the image the basis generators
  see; InstanceNorm in the basis stack includes it in its statistics. Only
  Drishti is non-square, so this is a Drishti-specific effect — which is one
  reason FOV gating is worth an ablation.
- **`domains` must stay complete** (four entries) in every config; the locked
  manifests are rebuilt from it during `check`. Participation is the protocol list.
- **Train-on-all requires `test_conditioning: "oracle"`** for any conditioned arm; LODO uses `"nearest_domain"`.
- **CREATE:** ssh hangs until MFA is refreshed at the KCL portal; `--requeue` is
  set but preempted jobs die — relaunch into the same `--out-dir` to resume from
  `resume_state.pt`; submit from `erc-hpc-login1/2`; `erc-hpc-comp223` has a
  faulty GPU and is excluded in the Step 4 scripts. Smoke wall time 20 min.
- **Do not launch a grid** before the smoke and a single seed-42 run have been
  read. My supervisor reads three-line messages: result, conclusion, next step.

## 7. Definition of done for Step 5, first milestone

- `SpatialFiLM` and `SpatialFiLMUNet` exist with the tests in Pieces 1–2 green;
  the K=0 identity test passes to 1e-6.
- Global FiLM and plain code paths are unchanged (the existing 449 tests pass
  untouched, and a Global FiLM fingerprint computed before and after is equal).
- The train-on-all K=8 config loads, `check` passes, the CPU smoke runs end to
  end with `conditioning.film.rank == 8`.
- One seed on CREATE and an aggregator table: SpFiLM vs plain, SpFiLM vs Global
  FiLM, with the wrong-code penalty next to it.

Start with Piece 1. Before giving me the contract, restate in your own words how
SpFiLM differs from the Global FiLM layer that already exists here and what the
K=0 identity means for the implementation — so I can check we agree before I
write anything.
