"""
常數定義：任務所需的標籤與類別。
"""

from itertools import product

# BIO 標籤
ASPECT_TAGS = ["O", "B-ASPECT", "I-ASPECT"]
OPINION_TAGS = ["O", "B-OPINION", "I-OPINION"]


# 領域類別
LAPTOP_ENTITIES = [
    "LAPTOP",
    "DISPLAY",
    "KEYBOARD",
    "MOUSE",
    "MOTHERBOARD",
    "CPU",
    "FANS_COOLING",
    "PORTS",
    "MEMORY",
    "POWER_SUPPLY",
    "OPTICAL_DRIVES",
    "BATTERY",
    "GRAPHICS",
    "HARD_DISK",
    "MULTIMEDIA_DEVICES",
    "HARDWARE",
    "SOFTWARE",
    "OS",
    "WARRANTY",
    "SHIPPING",
    "SUPPORT",
    "COMPANY",
]

LAPTOP_ATTRIBUTES = [
    "GENERAL",
    "PRICE",
    "QUALITY",
    "DESIGN_FEATURES",
    "OPERATION_PERFORMANCE",
    "USABILITY",
    "PORTABILITY",
    "CONNECTIVITY",
    "MISCELLANEOUS",
]

RESTAURANT_ENTITIES = ["RESTAURANT", "FOOD", "DRINKS", "AMBIENCE", "SERVICE", "LOCATION"]

RESTAURANT_ATTRIBUTES = ["GENERAL", "PRICES", "QUALITY", "STYLE_OPTIONS", "MISCELLANEOUS"]


def build_categories() -> list[str]:
    """建立所有合法的 ENTITY#ATTRIBUTE 組合。"""
    combos = set()
    combos.update(f"{ent}#{attr}" for ent, attr in product(LAPTOP_ENTITIES, LAPTOP_ATTRIBUTES))
    combos.update(f"{ent}#{attr}" for ent, attr in product(RESTAURANT_ENTITIES, RESTAURANT_ATTRIBUTES))
    return sorted(combos)


ALL_CATEGORIES = build_categories()
CATEGORY2ID = {label: idx for idx, label in enumerate(ALL_CATEGORIES)}
ID2CATEGORY = {idx: label for label, idx in CATEGORY2ID.items()}

