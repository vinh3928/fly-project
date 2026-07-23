"""Phase 4 exit-criteria constants for the multimodal mind pipeline."""

EMBEDDING_LATENT_DIM = 256
AFFECT_DIM = 16
TOTAL_EMBEDDING_DIM = EMBEDDING_LATENT_DIM + AFFECT_DIM  # 272

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

STM_TABLE = "fly_mind.stm_embeddings"
THOUGHTS_TABLE = "fly_mind.thoughts"
