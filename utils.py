# utils.py
def get_waste_info(label, confidence):
    info = {
        
        "biological": {
            "category": "Biodegradable Waste",
            "recyclable": False,
            "bin": "Green Bin",
            "tip": "Can be composted into natural fertilizer."
        },
        
        "glass": {
            "category": "Glass Waste",
            "recyclable": True,
            "bin": "Blue Bin",
            "tip": "Recycle bottles/jars. Clean before disposal."
        },
       
        "cardboard": {
            "category": "Cardboard Waste",
            "recyclable": True,
            "bin": "Blue Bin",
            "tip": "Flatten boxes before disposal to save space."
        },
        "clothes": {
            "category": "Textile Waste",
            "recyclable": False,
            "bin": "Green Bin",
            "tip": "Donate if reusable, otherwise dispose responsibly."
        },
        "shoes": {
            "category": "Footwear Waste",
            "recyclable": False,
            "bin": "Green Bin",
            "tip": "Donate if wearable, else dispose in general waste."
        },
        "Hazardous": {
            "category": "Hazardous Waste",
            "recyclable": False,
            "bin": "Special Collection",
            "tip": "Chemicals/paints must go to hazardous waste centers."
        },
        "medical": {
            "category": "Medical Waste",
            "recyclable": False,
            "bin": "Red Bin",
            "tip": "Use biomedical bins for syringes, medicines, and masks."
        },
        "metal": {
            "category": "Metal Waste",
            "recyclable": True,
            "bin": "Blue Bin",
            "tip": "Clean cans before disposal to improve recycling."
        },
        "paper": {
            "category": "Paper Waste",
            "recyclable": True,
            "bin": "Blue Bin",
            "tip": "Keep paper dry and free from food waste."
        },
        "plastic": {
            "category": "Plastic Waste",
            "recyclable": True,
            "bin": "Blue Bin",
            "tip": "Rinse bottles and containers before recycling."
        },
        "trash": {
            "category": "General Waste",
            "recyclable": False,
            "bin": "Green Bin",
            "tip": "Dispose responsibly in general waste."
        },
    }

    return {
        "waste_type": label,
        "confidence": round(confidence * 100, 2),
        "category": info.get(label, {}).get("category", "Unknown"),
        "recyclable": info.get(label, {}).get("recyclable", False),
        "bin": info.get(label, {}).get("bin", "Green Bin"),
        "tip": info.get(label, {}).get("tip", "Dispose responsibly."),
    }
