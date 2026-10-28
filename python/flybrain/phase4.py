"""Phase 4 exit-criteria constants for the multimodal mind pipeline."""

EMBEDDING_LATENT_DIM = 256
AFFECT_DIM = 16
TOTAL_EMBEDDING_DIM = EMBEDDING_LATENT_DIM + AFFECT_DIM  # 272

# Per-modality STM vector sizes (pgvector columns on short_term_memory).
EYE_VEC_DIM = 128
BODY_VEC_DIM = 128
CHEMICAL_VEC_DIM = 64
SENSORS_VEC_DIM = BODY_VEC_DIM + CHEMICAL_VEC_DIM  # 192
STATUS_VEC_DIM = 32
AFFECT_VEC_DIM = AFFECT_DIM  # 16
# Flattened stereo hexagonal-eye triangular facets (6 tris × 5 feats × 2 eyes).
PATCH_CELL_FEAT_DIM = 5
HEX_FACET_COUNT = 6  # pointy-top hex → center diagonals → 6 triangles
PATCH_GRID_CELLS = HEX_FACET_COUNT  # alias: facets per eye (was 3×3=9)
PATCH_VEC_DIM = PATCH_GRID_CELLS * PATCH_CELL_FEAT_DIM * 2  # 60
# Antenna L+R chemical channels (8 each) for interrupt smell history.
SMELL_VEC_DIM = 16
# Interrupt joint history: L2-normalized concat(patch, smell, sensors).
INTERRUPT_JOINT_VEC_DIM = PATCH_VEC_DIM + SMELL_VEC_DIM + SENSORS_VEC_DIM  # 268

HIGH_LEVEL_ACTION_DIM = 5
DEFAULT_OBS_DIM = 197

AFFECT_KEYS = (
    "surprise",
    "joy",
    "sad",
    "anger",
    "fear",
    "disgust",
    "trust",
    "anticipation",
    "altruism",
    "survival_risk",
    "energy_threat",
    "social_drive",
    "exploration",
    "comfort",
    "pain",
    "novelty",
)

STM_TABLE = "fly_mind.short_term_memory"
STM_TABLE_LEGACY = "fly_mind.stm_embeddings"
THOUGHTS_TABLE = "fly_mind.thoughts"

# Modality names accepted by ShortTermMemory.similar_by
STM_MODALITIES = (
    "embedding",
    "left_eye",
    "right_eye",
    "sensors",
    "status",
    "affect",
    "patches",
    "smell",
    "joint",
)
