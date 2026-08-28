# Define base colors
COLORS = [
    "black", "white", "grey", "red", "blue", "green", "yellow", 
    "orange", "purple", "pink", "brown", "navy", "beige", 
    "maroon", "olive", "teal", "mustard"
]

# Define base clothing items
TOPS = [
    "t-shirt", "shirt", "polo shirt", "button-down shirt", "tank top", 
    "crop top", "blouse", "tunic", "flannel shirt", "sweater", 
    "cardigan", "hoodie", "sweatshirt"
]

BOTTOMS = [
    "jeans", "pants", "shorts", "sweatpants", "leggings", "khaki pants", 
    "cargo pants", "cargo shorts", "skirt", "maxi skirt", "mini skirt", 
    "midi skirt", "pleated skirt"
]

OUTERWEAR = [
    "jacket", "coat", "windbreaker", "blazer", "trench coat", 
    "puffer jacket", "leather jacket", "denim jacket", "fleece jacket", 
    "poncho", "vest", "parka"
]

FULL_BODY = [
    "dress", "sundress", "evening gown", "suit", "tuxedo", "jumpsuit", "romper"
]

SHOES = [
    "sneakers", "boots", "shoes", "sandals", "high heels", "flats", "loafers"
]

# Generate every possible color combination (17 colors * ~50 items = ~850 categories)
CLOTHING_CATEGORIES = []
for color in COLORS:
    for item in TOPS + BOTTOMS + OUTERWEAR + FULL_BODY + SHOES:
        CLOTHING_CATEGORIES.append(f"{color} {item}")

# Add some specific styles that aren't just colors
CLOTHING_CATEGORIES.extend([
    "graphic t-shirt", "striped shirt", "plaid shirt", "floral dress", 
    "denim jeans", "leather pants", "yoga pants"
])

