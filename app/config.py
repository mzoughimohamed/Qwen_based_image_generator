from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"
STATIC_DIR = ROOT / "static"

MODEL_ID = "Qwen/Qwen-Image-2.1"
PE_T2I_ID = "Qwen/Qwen-Image-2.1-PE-T2I"
PE_I2I_ID = "Qwen/Qwen-Image-2.1-PE-I2I"
SPACE_ID = "Qwen/Qwen-Image-2.1"

RGBA_PHRASE = (
    "This is an RGBA image with transparency. "
    "The image has alpha channel and the background is transparent."
)

MAX_IMAGES = 10
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_STEPS, MAX_STEPS, DEFAULT_STEPS = 1, 100, 40
MAX_SEED = 2**32 - 1
CUSTOM_MIN, CUSTOM_MAX = 256, 2688
DEFAULT_T2I_SIZE = (2048, 2048)
ENHANCER_MAX_NEW_TOKENS = 4096

# (aspect label, width, height), matching the official Space presets.
SIZE_PRESETS = {
    "2K": [
        ("16:9", 2688, 1536),
        ("9:16", 1536, 2688),
        ("1:1", 2048, 2048),
        ("4:3", 2368, 1728),
        ("3:4", 1728, 2368),
    ],
    "1K": [
        ("16:9", 1344, 768),
        ("9:16", 768, 1344),
        ("1:1", 1024, 1024),
        ("4:3", 1184, 864),
        ("3:4", 864, 1184),
    ],
}
