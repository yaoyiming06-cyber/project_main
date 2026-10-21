"""Competition label order and study-level label conventions."""

LABEL_NAMES = (
    "Enlarged Cardiomediastinum",
    "Pneumothorax",
    "Consolidation",
    "Pneumonia",
    "Edema",
    "Cardiomegaly",
    "Atelectasis",
    "Lung Opacity",
    "Pleural Effusion",
    "No Finding",
)
LABEL_COLUMNS = tuple(f"label_{index}" for index in range(len(LABEL_NAMES)))
NUM_LABELS = len(LABEL_NAMES)

# Small-data specialist branches selected from the earlier label analysis.
HARD_LABEL_INDICES = (0, 3, 7)
HARD_LABEL_NAMES = tuple(LABEL_NAMES[index] for index in HARD_LABEL_INDICES)

VIEW_NAMES = ("AP", "PA", "LATERAL", "LL", "PA LLD", "UNKNOWN")
VIEW_TO_ID = {name: index for index, name in enumerate(VIEW_NAMES)}


def normalize_view_position(value):
    value = " ".join((value or "").strip().upper().split())
    return value if value in VIEW_TO_ID and value != "UNKNOWN" else "UNKNOWN"
