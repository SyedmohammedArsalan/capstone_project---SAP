import streamlit as st
from PIL import Image
import random
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from collections import Counter
import cv2
import os
import tempfile
import uuid
import base64
from io import BytesIO
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic import PredictionServiceClient
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize API clients
aiplatform.init(credentials=None, api_key=GEMINI_API_KEY)
openai.api_key = OPENAI_API_KEY

# ====================
# CONSTANTS & DATA
# ====================

STYLES = {
    "Casual": {"description": "Relaxed, comfortable clothing for everyday wear", "icon": "👕"},
    "Formal": {"description": "Professional attire for work or special occasions", "icon": "👔"},
    "Sporty": {"description": "Activewear for workouts and athletic activities", "icon": "🏃‍♀️"},
    "Bohemian": {"description": "Free-spirited, flowy, and artistic clothing", "icon": "🌿"},
    "Streetwear": {"description": "Urban-inspired fashion with bold statements", "icon": "🏙️"},
    "Minimalist": {"description": "Simple, clean lines with neutral colors", "icon": "⬜"},
    "Chic": {"description": "Elegant, sophisticated, and trendy", "icon": "💃"}
}

COLOR_PALETTES = {
    "Warm Tones": ["Beige", "Mustard", "Rust", "Olive", "Brown", "Terracotta", "Camel"],
    "Cool Tones": ["Navy", "Gray", "Burgundy", "Emerald", "Black", "Slate", "Charcoal"],
    "Bright Tones": ["Pink", "Cyan", "Turquoise", "Lime", "White", "Magenta", "Electric Blue"],
    "Pastel Tones": ["Lavender", "Mint", "Peach", "Baby Blue", "Powder Pink"],
    "Earth Tones": ["Sage Green", "Clay", "Sand", "Moss", "Taupe"]
}

OCCASIONS = ["Work", "Date Night", "Casual Outing", "Formal Event", "Vacation", "Workout", "Brunch"]
SEASONS = ["Spring", "Summer", "Fall", "Winter"]
GENDERS = ["Unisex", "Men", "Women"]
CLOTHING_TYPES = ["Shirt", "Pants", "Dress", "Skirt", "Jacket", "Sweater", "Shorts"]
PATTERNS = ["Solid", "Striped"]
SLEEVE_TYPES = ["Long-Sleeve", "Short-Sleeve"]

MOODS = {
    "Any": {"styles": list(STYLES.keys()), "palettes": list(COLOR_PALETTES.keys())},
    "Bold & Confident": {"styles": ["Streetwear", "Chic"], "palettes": ["Bright Tones", "Cool Tones"]},
    "Relaxed & Cozy": {"styles": ["Casual", "Minimalist"], "palettes": ["Earth Tones", "Warm Tones"]},
    "Elegant & Timeless": {"styles": ["Formal", "Chic", "Minimalist"], "palettes": ["Cool Tones", "Pastel Tones"]}
}

PRODUCT_LINKS = {
    "Casual": {
        "Men": {
            "Shirt": [
                {"url": "https://www.hm.com/product/mens-cotton-shirt-beige", "name": "H&M Cotton Shirt", "color": "Beige", "price": "$24.99"},
                {"url": "https://www.zara.com/us/en/denim-shirt-navy-p1234", "name": "Zara Denim Shirt", "color": "Navy", "price": "$39.99"}
            ],
            "Pants": [
                {"url": "https://www.uniqlo.com/us/en/mens-chinos-khaki-p5678", "name": "Uniqlo Chinos", "color": "Khaki", "price": "$49.90"}
            ]
        },
        "Women": {
            "Dress": [
                {"url": "https://www.asos.com/womens-midi-dress-mint-4567", "name": "ASOS Midi Dress", "color": "Mint", "price": "$55.00"}
            ]
        },
        "Unisex": {
            "Shirt": [
                {"url": "https://www.everlane.com/products/unisex-tee-white-1234", "name": "Everlane Organic Tee", "color": "White", "price": "$28.00"}
            ]
        }
    },
    "Formal": {
        "Men": {
            "Shirt": [
                {"url": "https://www.brooksbrothers.com/mens-dress-shirt-white-5678", "name": "Brooks Brothers Dress Shirt", "color": "White", "price": "$89.50"}
            ]
        },
        "Women": {
            "Dress": [
                {"url": "https://www.reiss.com/womens-sheath-dress-black-3456", "name": "Reiss Sheath Dress", "color": "Black", "price": "$199.00"}
            ]
        }
    },
    "Streetwear": {
        "Unisex": {
            "Shirt": [
                {"url": "https://www.supreme.com/mens-graphic-tee-white-5678", "name": "Supreme Graphic Tee", "color": "White", "price": "$48.00"}
            ]
        }
    },
    "Chic": {
        "Women": {
            "Dress": [
                {"url": "https://www.reiss.com/womens-wrap-dress-burgundy-2345", "name": "Reiss Wrap Dress", "color": "Burgundy", "price": "$179.00"}
            ]
        }
    },
    "Minimalist": {
        "Unisex": {
            "Shirt": [
                {"url": "https://www.cos.com/mens-tee-white-1234", "name": "COS Basic Tee", "color": "White", "price": "$35.00"}
            ]
        }
    }
}

TREND_DATA = {
    "Boho Chic": {"score": 9, "description": "Streamlined bohemian with suede and embroidery"},
    "Maximalism": {"score": 8, "description": "Bold colors and mixed textures"},
    "Athleisure": {"score": 7, "description": "Elevated sportswear for daily wear"},
    "Sheer Fabrics": {"score": 6, "description": "Diaphanous, romantic layers"},
    "Rich Neutrals": {"score": 6, "description": "Deep browns, beiges, and burgundies"}
}

PRE_CURATED_OUTFITS = [
    {
        "name": "Boho Chic Maxi Dress Ensemble",
        "description": "Flowy, embroidered maxi dress in sage green, paired with tan suede ankle boots.",
        "style_tips": "Add a silk headscarf and layered necklaces.",
        "image_url": "https://www.anthropologie.com/shop/boho-maxi-dress",
        "trend": "Boho Chic"
    },
    {
        "name": "Maximalist Animal Print Statement",
        "description": "Leopard print midi skirt with a mustard yellow silk blouse.",
        "style_tips": "Balance the print with solid colors.",
        "image_url": "https://www.zara.com/us/en/leopard-midi-skirt-p1234",
        "trend": "Maximalism"
    }
]

# ====================
# WARDROBE MANAGEMENT
# ====================

def add_wardrobe_item(image, clothing_type, color):
    if "wardrobe" not in st.session_state:
        st.session_state.wardrobe = []
    item = {
        "id": str(uuid.uuid4()),
        "image": image,
        "type": clothing_type,
        "color": color,
        "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.wardrobe.append(item)
    update_user_progress("add_wardrobe", 15)
    return item

def delete_wardrobe_item(item_id):
    if "wardrobe" in st.session_state:
        st.session_state.wardrobe = [item for item in st.session_state.wardrobe if item["id"] != item_id]

# ====================
# GAMIFICATION
# ====================

def update_user_progress(action, points):
    if "user_progress" not in st.session_state:
        st.session_state.user_progress = {"points": 0, "badges": []}
    
    st.session_state.user_progress["points"] += points
    
    badges = [
        {"name": "Style Novice", "points": 50, "icon": "🌟"},
        {"name": "Fashionista", "points": 100, "icon": "👗"},
        {"name": "Wardrobe Wizard", "condition": len(st.session_state.get("wardrobe", [])) >= 5, "icon": "🧳"}
    ]
    
    for badge in badges:
        if badge.get("points", float("inf")) <= st.session_state.user_progress["points"] and badge["name"] not in st.session_state.user_progress["badges"]:
            st.session_state.user_progress["badges"].append(badge["name"])
        elif badge.get("condition", False) and badge["name"] not in st.session_state.user_progress["badges"]:
            st.session_state.user_progress["badges"].append(badge["name"])

# ====================
# SENTIMENT ANALYSIS
# ====================

def simulate_x_posts(style):
    templates = [
        f"This {style} vibe is 🔥! Loving the bold choices!",
        f"Wow, {style} looks amazing! Perfect for a night out!",
        f"Not sure about {style}, feels a bit too much for me.",
        f"{style} is so chic! Great color palette!",
        f"Meh, {style} isn't my thing, too plain.",
        f"Obsessed with this {style} look! So stylish!"
    ]
    return random.sample(templates, min(4, len(templates)))

def analyze_sentiment(posts):
    positive_keywords = ["love", "amazing", "great", "perfect", "chic", "stylish", "obsessed", "🔥"]
    negative_keywords = ["not", "too much", "meh", "plain", "bad"]
    sentiments = []
    
    for post in posts:
        post_lower = post.lower()
        if any(keyword in post_lower for keyword in positive_keywords):
            sentiments.append("Positive")
        elif any(keyword in post_lower for keyword in negative_keywords):
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")
    
    return Counter(sentiments)

# ====================
# IMAGE PROCESSING
# ====================

def gemini_analyze_image(image):
    try:
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Configure Gemini API call
        project_id = "your-project-id"  # Replace with your Google Cloud project ID
        endpoint_id = "your-endpoint-id"  # Replace with your Vertex AI endpoint ID
        location = "us-central1"
        endpoint = f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"
        
        client = PredictionServiceClient(client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"})
        instance = {
            "image": {"content": image_bytes},
            "parameters": {"confidenceThreshold": 0.8}
        }
        response = client.predict(endpoint=endpoint, instances=[instance])
        
        # Parse Gemini response
        prediction = response.predictions[0]
        result = {
            "items": [
                {
                    "type": prediction.get("type", "Shirt"),
                    "color": prediction.get("color", "Navy"),
                    "pattern": prediction.get("pattern", "Striped"),
                    "confidence": prediction.get("confidence", 0.8)
                }
            ],
            "description": prediction.get("description", f"A {prediction.get('color', 'Navy')} {prediction.get('pattern', 'Striped')} {prediction.get('type', 'Shirt')}.")
        }
        update_user_progress("gemini_analysis", 10)
        return result
    except Exception as e:
        st.warning(f"Gemini API failed: {str(e)}. Falling back to default analysis.")
        return None

def openai_generate_style_suggestions(gemini_output, num_suggestions=3):
    try:
        item = gemini_output["items"][0]
        prompt = f"Suggest {num_suggestions} stylish outfits for a {item['color']} {item['pattern']} {item['type']} in a casual setting. Provide concise suggestions, each starting with 'Pair with'."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a fashion expert providing concise outfit suggestions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            n=1,
            temperature=0.7
        )
        suggestions = response.choices[0].message.content.strip().split("\n")[:num_suggestions]
        if not suggestions or len(suggestions) < num_suggestions:
            suggestions = [
                f"Pair with khaki chinos and white sneakers.",
                f"Pair with dark jeans and loafers.",
                f"Pair with a beige skirt and ankle boots."
            ][:num_suggestions]
        st.session_state.openai_suggestions = suggestions
        update_user_progress("openai_suggestions", 10)
        return suggestions
    except Exception as e:
        st.warning(f"OpenAI API failed: {str(e)}. Using default suggestions.")
        return [
            f"Pair with khaki chinos and white sneakers.",
            f"Pair with dark jeans and loafers.",
            f"Pair with a beige skirt and ankle boots."
        ][:num_suggestions]

def extract_dominant_colors(image, num_colors=5):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    clt = KMeans(n_clusters=num_colors, n_init=10)
    labels = clt.fit_predict(img)
    label_counts = Counter(labels)
    dominant_colors = [clt.cluster_centers_[i] for i in label_counts.keys()]
    hex_colors = [rgb_to_hex(color) for color in dominant_colors]
    
    gray = cv2.cvtColor(cv2.resize(np.array(image), (100, 100)), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
    pattern = "Striped" if lines is not None and len(lines) > 5 else "Solid"
    
    return hex_colors, pattern

def rgb_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def detect_clothing_items(image, user_clothing_types=None):
    dominant_colors, pattern = extract_dominant_colors(image)
    color_to_clothing = {
        "Bright Tones": ["Shirt", "Dress", "Shorts"],
        "Pastel Tones": ["Shirt", "Dress", "Skirt"],
        "Warm Tones": ["Sweater", "Pants", "Jacket"],
        "Cool Tones": ["Pants", "Jacket", "Shirt"],
        "Earth Tones": ["Pants", "Sweater", "Jacket"]
    }
    palette_name = next((name for name, colors in COLOR_PALETTES.items() if any(c.lower() in [col.lower() for col in colors] for c in dominant_colors)), "Cool Tones")
    possible_clothing = color_to_clothing.get(palette_name, CLOTHING_TYPES)
    
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(cv2.Canny(gray, 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clothing_details = []
    confidence = 0.8
    
    for contour in contours[:2]:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        clothing_type = random.choice(possible_clothing)
        if clothing_type == "Shirt":
            sleeve = "Long-Sleeve" if aspect_ratio < 0.5 else "Short-Sleeve"
            clothing_details.append(f"{sleeve} {clothing_type}")
            confidence = 0.85 if aspect_ratio < 0.5 else 0.75
        else:
            clothing_details.append(clothing_type)
    
    if user_clothing_types:
        detected = [item for item in user_clothing_types if item in possible_clothing]
        if detected:
            return detected[:2], pattern, confidence
    return clothing_details[:2] or ["Shirt"], pattern, confidence

# ====================
# VIRTUAL TRY-ON & AR
# ====================

def render_try_on_preview(analysis, ar_mode=False):
    gender = analysis["gender"]
    clothing_items = analysis["detected_clothing"] or analysis["clothing_types"] or ["Shirt"]
    colors = analysis["palette_colors"][:2] or ["#ffffff", "#000000"]
    wardrobe_items = analysis.get("wardrobe_items", [])
    
    canvas_width = 200
    canvas_height = 400
    canvas_id = f"tryOnCanvas_{uuid.uuid4()}"
    canvas = f"""
    <div style='text-align: center;'>
        <canvas id='{canvas_id}' width='{canvas_width}' height='{canvas_height}'></canvas>
        <p style='color: #ff5555; font-size: 0.9em; display: none;' id='canvasError_{canvas_id}'>Canvas rendering failed. Check browser console.</p>
    </div>
    <script>
    function drawAvatar() {{
        console.log('Drawing canvas {canvas_id}');
        try {{
            var canvas = document.getElementById('{canvas_id}');
            if (!canvas || !canvas.getContext) {{
                document.getElementById('canvasError_{canvas_id}').style.display = 'block';
                console.error('Canvas not supported or not found');
                return;
            }}
            var ctx = canvas.getContext('2d');

            ctx.fillStyle = '#e0e0e0';
            ctx.beginPath();
            ctx.ellipse(100, 50, 30, 30, 0, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillRect(80, 80, 40, 120);
            ctx.fillRect(80, 200, 20, 100);
            ctx.fillRect(100, 200, 20, 100);
            ctx.fillRect(60, 80, 20, 60);
            ctx.fillRect(120, 80, 20, 60);
    """

    if clothing_items:
        for i, item in enumerate(clothing_items[:2]):
            color = colors[i % len(colors)]
            wardrobe_item = next((w for w in wardrobe_items if w["type"] in item), None)
            alpha = 0.7 if ar_mode else 1.0
            canvas += f"""
            ctx.globalAlpha = {alpha};
            console.log('Drawing item: {item}, alpha: {alpha}');
            """
            if wardrobe_item:
                canvas += f"""
                ctx.fillStyle = '{wardrobe_item["color"]}';
                """
            else:
                canvas += f"""
                ctx.fillStyle = '{color}';
                """
            
            if "Shirt" in item:
                canvas += "ctx.fillRect(70, 80, 60, 100);"
            elif "Pants" in item:
                canvas += "ctx.fillRect(80, 200, 40, 100);"
            elif "Dress" in item:
                canvas += """
                ctx.beginPath();
                ctx.moveTo(70, 80);
                ctx.lineTo(130, 80);
                ctx.lineTo(150, 200);
                ctx.lineTo(50, 200);
                ctx.closePath();
                ctx.fill();
                """
            elif "Skirt" in item:
                canvas += """
                ctx.beginPath();
                ctx.moveTo(70, 160);
                ctx.lineTo(130, 160);
                ctx.lineTo(150, 200);
                ctx.lineTo(50, 200);
                ctx.closePath();
                ctx.fill();
                """
            elif "Jacket" in item:
                canvas += "ctx.fillRect(60, 80, 80, 120);"
            elif "Sweater" in item:
                canvas += "ctx.fillRect(60, 80, 80, 100);"
            elif "Shorts" in item:
                canvas += "ctx.fillRect(80, 200, 40, 50);"
            canvas += "ctx.globalAlpha = 1.0;"

    canvas += """
        }} catch (e) {{
            console.error('Canvas error:', e);
            document.getElementById('canvasError_{canvas_id}').style.display = 'block';
        }}
    }}
    console.log('Scheduling canvas draw for {canvas_id}');
    if (document.readyState === 'complete') {{
        drawAvatar();
    }} else {{
        window.addEventListener('load', drawAvatar);
    }}
    </script>
    """
    return canvas

# ====================
# ANALYSIS FUNCTIONS
# ====================

def analyze_fashion_style(image, occasion=None, season=None, gender="Unisex", clothing_types=None):
    try:
        # Try Gemini for advanced analysis
        gemini_result = gemini_analyze_image(image)
        if gemini_result:
            detected_clothing = [item["type"] for item in gemini_result["items"]]
            pattern = gemini_result["items"][0]["pattern"]
            confidence = gemini_result["items"][0]["confidence"]
            gemini_description = gemini_result["description"]
            colors = [gemini_result["items"][0]["color"]]
        else:
            dominant_colors, pattern = extract_dominant_colors(image)
            detected_clothing, pattern, confidence = detect_clothing_items(image, clothing_types)
            gemini_description = None
            colors = dominant_colors

        # Generate OpenAI style suggestions
        if gemini_description:
            openai_suggestions = openai_generate_style_suggestions(gemini_result)
        else:
            openai_suggestions = []

        mood = st.session_state.get("mood", "Any")
        possible_styles = MOODS[mood]["styles"]
        possible_palettes = MOODS[mood]["palettes"]

        wardrobe = st.session_state.get("wardrobe", [])
        wardrobe_items = []
        if wardrobe and (clothing_types or detected_clothing):
            target_types = list(set((clothing_types or []) + [c.split()[-1] for c in detected_clothing]))
            for item in wardrobe:
                if item["type"] in target_types and item["color"] in COLOR_PALETTES.get(next(iter(COLOR_PALETTES)), []):
                    wardrobe_items.append(item)
            detected_clothing = [item["type"] for item in wardrobe_items[:2]] or detected_clothing

        styles = possible_styles
        if occasion == "Work":
            styles = [s for s in possible_styles if s in ["Formal", "Chic", "Minimalist"]]
        elif occasion == "Date Night":
            styles = [s for s in possible_styles if s in ["Chic", "Bohemian", "Formal"]]
        elif occasion == "Workout":
            styles = [s for s in possible_styles if s == "Sporty"]
        
        color_palettes = possible_palettes
        if season == "Winter":
            color_palettes = [p for p in possible_palettes if p in ["Warm Tones", "Cool Tones", "Earth Tones"]]
        elif season == "Summer":
            color_palettes = [p for p in possible_palettes if p in ["Bright Tones", "Pastel Tones"]]

        if not styles:
            styles = possible_styles
        if not color_palettes:
            color_palettes = possible_palettes

        selected_style = random.choice(styles)
        selected_palette = random.choice(color_palettes)
        palette_colors = COLOR_PALETTES[selected_palette]
        accessories = recommend_accessories(selected_style, gender)
        update_user_progress("analyze_improved", 10)
        return {
            "style": selected_style,
            "style_description": STYLES[selected_style]["description"],
            "style_icon": STYLES[selected_style]["icon"],
            "palette_name": selected_palette,
            "palette_colors": colors or palette_colors,
            "dominant_colors": colors or palette_colors,
            "pattern": pattern,
            "accessories": accessories,
            "occasion": occasion,
            "season": season,
            "gender": gender,
            "clothing_types": clothing_types or [],
            "detected_clothing": detected_clothing,
            "mood": mood,
            "wardrobe_items": wardrobe_items[:2],
            "detection_confidence": confidence,
            "gemini_description": gemini_description,
            "openai_suggestions": openai_suggestions
        }
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None

def recommend_accessories(style, gender):
    accessories = {
        "Casual": {
            "Men": ["Casual Watch", "Leather Belt", "Canvas Backpack"],
            "Women": ["Minimal Necklace", "Tote Bag", "Sunglasses"],
            "Unisex": ["Minimal Jewelry", "Casual Watch", "Canvas Tote"]
        },
        "Formal": {
            "Men": ["Dress Watch", "Cufflinks", "Leather Briefcase"],
            "Women": ["Pearl Earrings", "Silk Scarf", "Clutch"],
            "Unisex": ["Leather Briefcase", "Dress Watch", "Silk Scarf"]
        },
        "Streetwear": {
            "Unisex": ["Bucket Hat", "Chain Necklace", "Crossbody Bag"]
        },
        "Chic": {
            "Women": ["Statement Earrings", "Designer Clutch", "Heeled Sandals"],
            "Unisex": ["Statement Earrings", "Clutch", "Designer Sunglasses"]
        },
        "Minimalist": {
            "Unisex": ["Simple Studs", "Sleek Watch", "Structured Tote"]
        }
    }
    return random.sample(accessories.get(style, {}).get(gender, accessories.get(style, {}).get("Unisex", [])), 3)

def generate_outfit_ideas(analysis_result, num_outfits=3):
    style = analysis_result["style"]
    gender = analysis_result["gender"]
    clothing_types = analysis_result["clothing_types"]
    detected_clothing = analysis_result["detected_clothing"]
    palette = analysis_result["palette_name"]
    wardrobe_items = analysis_result.get("wardrobe_items", [])
    pattern = analysis_result["pattern"]
    openai_suggestions = analysis_result.get("openai_suggestions", [])

    outfits = openai_suggestions[:num_outfits]
    if len(outfits) < num_outfits:
        base_ideas = {
            "Casual": {
                "Men": [
                    {"type": "Shirt", "desc": f"{pattern} slim-fit cotton shirt with chinos and sneakers"},
                    {"type": "Pants", "desc": "Denim jeans with a plain tee and loafers"}
                ],
                "Women": [
                    {"type": "Dress", "desc": f"{pattern} casual midi dress with sneakers and a denim jacket"}
                ],
                "Unisex": [
                    {"type": "Shirt", "desc": f"{pattern} loose-fit tee with relaxed chinos and sneakers"}
                ]
            },
            "Formal": {
                "Men": [
                    {"type": "Shirt", "desc": f"{pattern} dress shirt with tailored trousers and oxfords"}
                ],
                "Women": [
                    {"type": "Dress", "desc": f"{pattern} sheath dress with pumps and a blazer"}
                ]
            },
            "Streetwear": {
                "Unisex": [
                    {"type": "Shirt", "desc": f"{pattern} graphic tee with cargo pants and chunky sneakers"}
                ]
            },
            "Chic": {
                "Women": [
                    {"type": "Dress", "desc": f"{pattern} wrap dress with heeled boots and a statement belt"}
                ]
            },
            "Minimalist": {
                "Unisex": [
                    {"type": "Shirt", "desc": f"{pattern} monochrome tee with straight-leg pants and sneakers"}
                ]
            }
        }

        available_outfits = []
        if wardrobe_items:
            for item in wardrobe_items:
                desc = f"{pattern} {item['type']} in {item['color']}"
                available_outfits.append({"type": item["type"], "desc": desc})
        
        if not available_outfits:
            available_outfits = base_ideas.get(style, {}).get(gender, base_ideas.get(style, {}).get("Unisex", []))
            if clothing_types or detected_clothing:
                target_types = list(set(clothing_types + [c.split()[-1] for c in detected_clothing]))
                available_outfits = [outfit for outfit in available_outfits if outfit["type"] in target_types]
        
        if not available_outfits:
            available_outfits = base_ideas.get(style, {}).get(gender, base_ideas.get(style, {}).get("Unisex", []))
        
        selected_outfits = random.sample(available_outfits, min(num_outfits - len(outfits), len(available_outfits)))
        outfits.extend([f"{outfit['desc']}" for outfit in selected_outfits])
    
    complementary_outfits = []
    for clothing in detected_clothing:
        clothing_type = clothing.split()[-1]
        if clothing_type == "Shirt":
            complementary_outfits.append(f"Pair with tailored pants in {random.choice(analysis_result['palette_colors'])}")
        elif clothing_type == "Dress":
            complementary_outfits.append(f"Pair with a jacket in {random.choice(analysis_result['palette_colors'])}")
    
    return outfits[:num_outfits] + complementary_outfits[:2]

def recommend_stores(style, budget="Medium", gender="Unisex", gemini_color=None):
    stores = {
        "Casual": {
            "Men": {"Low": ["H&M", "Uniqlo"], "Medium": ["Gap", "J.Crew"], "High": ["Everlane"]},
            "Women": {"Low": ["H&M", "Zara"], "Medium": ["Madewell"], "High": ["Reformation"]},
            "Unisex": {"Low": ["H&M"], "Medium": ["Gap"], "High": ["Everlane"]}
        },
        "Formal": {
            "Men": {"Low": ["H&M"], "Medium": ["J.Crew"], "High": ["Theory"]},
            "Women": {"Low": ["Zara"], "Medium": ["Ann Taylor"], "High": ["Reiss"]}
        },
        "Streetwear": {
            "Unisex": {"Low": ["H&M"], "Medium": ["Urban Outfitters"], "High": ["Supreme"]}
        },
        "Chic": {
            "Women": {"Low": ["Zara"], "Medium": ["Reiss"], "High": ["Dior"]},
            "Unisex": {"Medium": ["Sandro"], "High": ["Chanel"]}
        },
        "Minimalist": {
            "Unisex": {"Low": ["COS"], "Medium": ["Arket"], "High": ["The Row"]}
        }
    }
    available_stores = stores.get(style, {}).get(gender, stores.get(style, {}).get("Unisex", {})).get(budget, ["Various retailers"])
    if gemini_color and gemini_color.lower() in ["navy", "navy blue"]:
        available_stores = [s for s in available_stores if s in ["H&M", "Zara", "J.Crew", "Everlane"]]
    return available_stores

def get_product_links(analysis_result, num_links=3):
    style = analysis_result["style"]
    gender = analysis_result["gender"]
    clothing_types = analysis_result["clothing_types"]
    detected_clothing = analysis_result["detected_clothing"]
    palette_colors = analysis_result["palette_colors"]
    gemini_color = analysis_result.get("gemini_description", "").split(",")[0].split(":")[-1].strip() if analysis_result.get("gemini_description") else None

    target_types = list(set(clothing_types + [c.split()[-1] for c in detected_clothing]))
    available_products = []
    
    for clothing_type in target_types:
        products = PRODUCT_LINKS.get(style, {}).get(gender, PRODUCT_LINKS.get(style, {}).get("Unisex", {})).get(clothing_type, [])
        for product in products:
            if gemini_color and gemini_color.lower() in product["color"].lower():
                available_products.append(product)
            elif product["color"].lower() in [c.lower() for c in palette_colors]:
                available_products.append(product)
    
    return random.sample(available_products, min(num_links, len(available_products))) if available_products else []

def save_recommendations(analysis, outfits, stores, product_links):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "timestamp": timestamp,
        "style": analysis['style'],
        "palette": analysis['palette_name'],
        "mood": analysis['mood'],
        "outfits": outfits,
        "stores": stores,
        "product_links": product_links
    }
    update_user_progress("save_recommendations", 5)
    return True

# ====================
# STREAMLIT APP
# ====================

def main():
    st.set_page_config(page_title="FashionAI", layout="wide", page_icon="👗")

    # Custom CSS for new theme
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&family=Lora:wght@400;700&display=swap');
        body {
            background: linear-gradient(135deg, #FFE4E1 0%, #F8F9FA 100%);
            font-family: 'Poppins', sans-serif;
            color: #2B2B2B;
        }
        .stApp {
            max-width: 1200px;
            margin: 20px auto;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            background: #FFFFFF;
            padding: 20px;
        }
        h1, h2, h3 {
            font-family: 'Lora', serif;
            color: #2B2B2B;
        }
        .stButton>button {
            background: linear-gradient(90deg, #FFD700, #FFC107);
            color: #2B2B2B;
            border-radius: 25px;
            padding: 12px 24px;
            border: none;
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #FFC107, #FFD700);
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .card {
            background: #F8F9FA;
            padding: 20px;
            border-radius: 15px;
            margin: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            text-align: center;
            opacity: 0;
            animation: fadeIn 0.5s ease forwards;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        }
        .banner {
            width: 100%;
            height: 300px;
            background: url('https://images.unsplash.com/photo-1529139574466-a303027c1d8b') center/cover no-repeat;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #FFFFFF;
            text-shadow: 0 2px 4px rgba(0,0,0,0.5);
            font-family: 'Lora', serif;
            font-size: 2.8em;
            margin-bottom: 20px;
        }
        .stExpander {
            border: 1px solid #FFE4E1;
            border-radius: 10px;
            margin-bottom: 10px;
            transition: all 0.3s ease;
        }
        .stExpander:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .stSelectbox>div>div, .stMultiSelect>div>div {
            border-radius: 10px;
            border: 1px solid #FFD700;
            transition: all 0.3s ease;
        }
        .stSelectbox>div>div:hover, .stMultiSelect>div>div:hover {
            border-color: #FFC107;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        canvas {
            border: 1px solid #FFE4E1;
            border-radius: 10px;
            margin: 10px auto;
            display: block;
        }
        .progress-bar {
            width: 100%;
            background-color: #FFE4E1;
            border-radius: 10px;
            overflow: hidden;
        }
        .progress-fill {
            height: 20px;
            background: linear-gradient(90deg, #FFD700, #FFC107);
            transition: width 0.5s ease;
        }
        .badge {
            display: inline-block;
            background: linear-gradient(90deg, #FFD700, #FFC107);
            color: #2B2B2B;
            padding: 6px 12px;
            border-radius: 15px;
            margin: 5px;
            font-size: 0.9em;
            font-weight: 600;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar: User Progress
    st.sidebar.title("🎮 Your Style Journey")
    progress = st.session_state.get("user_progress", {"points": 0, "badges": []})
    points = progress["points"]
    badges = progress["badges"]
    st.sidebar.markdown(f"**Points**: {points}")
    max_points = 150
    progress_percent = min((points / max_points) * 100, 100)
    st.sidebar.markdown(
        f"""
        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress_percent}%;"></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if badges:
        st.sidebar.markdown("**Badges**:")
        for badge in badges:
            st.sidebar.markdown(f'<span class="badge">{badge}</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown("No badges yet! Style more to earn some! 🌟")

    # Dynamic Banner
    st.markdown("""
        <div class="banner">
            Discover Your Signature Style ✨
        </div>
    """, unsafe_allow_html=True)

    st.title("👗 FashionAI - Your Style Muse")
    st.markdown("Capture your look, visualize outfits in AR, and explore 2025 trends with AI-powered recommendations!")

    # Initialize session state
    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    if "image" not in st.session_state:
        st.session_state.image = None
    if "detected_clothing" not in st.session_state:
        st.session_state.detected_clothing = []
    if "mood" not in st.session_state:
        st.session_state.mood = "Any"
    if "wardrobe" not in st.session_state:
        st.session_state.wardrobe = []
    if "user_progress" not in st.session_state:
        st.session_state.user_progress = {"points": 0, "badges": []}
    if "ar_mode" not in st.session_state:
        st.session_state.ar_mode = False
    if "detected_pattern" not in st.session_state:
        st.session_state.detected_pattern = "Solid"
    if "detection_confidence" not in st.session_state:
        st.session_state.detection_confidence = 0.8
    if "openai_suggestions" not in st.session_state:
        st.session_state.openai_suggestions = []

    # Tabs
    wardrobe_tab, analysis_tab, trends_tab = st.tabs(["👗 Wardrobe", "🔍 Analysis", "🌟 Trends"])

    with wardrobe_tab:
        st.subheader("🧳 Your Virtual Wardrobe")
        st.markdown("Build your collection and create personalized outfits!")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Add New Item")
            wardrobe_image = st.file_uploader("Upload Clothing Item", type=["jpg", "jpeg", "png"], key="wardrobe_uploader")
            clothing_type = st.selectbox("Clothing Type", CLOTHING_TYPES, key="wardrobe_type")
            color_options = [color for palette in COLOR_PALETTES.values() for color in palette]
            clothing_color = st.selectbox("Color", color_options, key="wardrobe_color")
            if st.button("Add to Wardrobe", key="add_wardrobe"):
                if wardrobe_image:
                    image = Image.open(wardrobe_image)
                    add_wardrobe_item(image, clothing_type, clothing_color)
                    st.success("Item added to wardrobe! +15 points 🌟")
                else:
                    st.error("Please upload an image.")
        
        with col2:
            st.markdown("### Your Wardrobe")
            if st.session_state.wardrobe:
                cols = st.columns(3)
                for i, item in enumerate(st.session_state.wardrobe):
                    with cols[i % 3]:
                        st.markdown(
                            f"""
                            <div class='card'>
                                <p><strong>{item['type']}</strong> ({item['color']})</p>
                                <p>Added: {item['upload_date']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.image(item["image"], width=100)
                        if st.button("Delete", key=f"delete_{item['id']}"):
                            delete_wardrobe_item(item["id"])
                            st.experimental_rerun()
            else:
                st.write("Your wardrobe is empty. Add some items to get started!")

    with analysis_tab:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("📸 Capture or Upload Your Look")
            st.markdown("Snap a photo or upload an image to get personalized style recommendations!")
            camera_image = st.camera_input("Take a Photo", key="camera_input")
            uploaded_file = st.file_uploader("Or Upload an Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            
            if camera_image:
                st.session_state.image = Image.open(camera_image)
                st.image(st.session_state.image, caption="Your captured look", use_column_width=True)
                update_user_progress("capture_image", 10)
            elif uploaded_file:
                st.session_state.image = Image.open(uploaded_file)
                st.image(st.session_state.image, caption="Your uploaded look", use_column_width=True)
                update_user_progress("upload_image", 10)

            if st.session_state.image:
                st.subheader("🔍 Refine Your Analysis")
                st.session_state.mood = st.selectbox("🌟 What's your mood today?", list(MOODS.keys()), index=0)
                gender = st.selectbox("Gender", GENDERS, index=0)
                clothing_types = st.multiselect("Preferred Clothing Types (Optional)", CLOTHING_TYPES, default=[])
                occasion = st.selectbox("Occasion", ["Any"] + OCCASIONS, index=0)
                season = st.selectbox("Season", ["Any"] + SEASONS, index=0)
                budget = st.select_slider("Budget Range", options=["Low", "Medium", "High"], value="Medium")
                st.session_state.occasion = occasion if occasion != "Any" else None
                st.session_state.season = season if season != "Any" else None
                st.session_state.gender = gender
                st.session_state.clothing_types = clothing_types

                with st.spinner("Detecting clothing items..."):
                    detected_clothing, pattern, confidence = detect_clothing_items(st.session_state.image, clothing_types)
                    st.session_state.detected_clothing = detected_clothing
                    st.session_state.detected_pattern = pattern
                    st.session_state.detection_confidence = confidence

                st.markdown("### 🧼 Detected Clothing Items")
                st.write(f"We detected these items (Pattern: {pattern}, Confidence: {confidence:.2f}):")
                clothing_options = []
                for p in PATTERNS:
                    for s in ["Long-Sleeve", "Short-Sleeve", ""]:
                        for ct in CLOTHING_TYPES:
                            if s and ct == "Shirt":
                                clothing_options.append(f"{p} {s} {ct}")
                            elif not s:
                                clothing_options.append(f"{p} {ct}")
                clothing_options += CLOTHING_TYPES
                normalized_detected = []
                for item in detected_clothing:
                    if item in clothing_options:
                        normalized_detected.append(item)
                    elif item.split()[-1] in CLOTHING_TYPES:
                        normalized_detected.append(item.split()[-1])
                confirmed_clothing = st.multiselect(
                    "Confirm or Adjust Detected Clothing",
                    clothing_options,
                    default=normalized_detected
                )

                if st.button("Analyze My Style", type="primary"):
                    with st.spinner("Analyzing your fashion style with AI..."):
                        analysis = analyze_fashion_style(
                            st.session_state.image,
                            occasion if occasion != "Any" else None,
                            season if season != "Any" else None,
                            gender,
                            confirmed_clothing or clothing_types
                        )
                        st.session_state.analysis = analysis
                        st.session_state.budget = budget
                        if analysis:
                            st.success(f"Styled for your {analysis['mood']} vibe! 🌟")
                            update_user_progress("analyze_style", 20)

        with col2:
            if st.session_state.analysis:
                analysis = st.session_state.analysis
                st.subheader(f"{analysis['style_icon']} Your Style Analysis")
                st.markdown(f"**{analysis['style']}** — {analysis['style_description']}")
                st.markdown(f"**Mood**: {analysis['mood']}")
                st.markdown(f"**Pattern**: {analysis['pattern']}")
                st.markdown(f"**Gender**: {analysis['gender']}")
                if analysis['clothing_types']:
                    st.markdown(f"**Preferred Clothing Types**: {', '.join(analysis['clothing_types'])}")
                if analysis['detected_clothing']:
                    st.markdown(f"**Detected Clothing**: {', '.join(analysis['detected_clothing'])} (Confidence: {analysis['detection_confidence']:.2f})")
                if analysis.get('gemini_description'):
                    st.markdown(f"**AI Analysis**: {analysis['gemini_description']}")

                st.markdown("### 👗 Virtual Try-On")
                st.markdown(f"See your look come to life! AR Mode: {'ON' if st.session_state.ar_mode else 'OFF'}")
                st.markdown("*Note: AR is simulated. For real AR, use AR.js/WebXR.*")
                st.checkbox("AR Mode", value=st.session_state.ar_mode, key="ar_mode", on_change=lambda: st.session_state.update({"ar_mode": not st.session_state.ar_mode}))
                if st.session_state.ar_mode:
                    update_user_progress("ar_mode", 15)
                st.markdown(
                    f"""
                    <div class='card'>
                        {render_try_on_preview(analysis, st.session_state.ar_mode)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    """
                    <div style='text-align: center; margin-top: 10px;'>
                        <a href='#' style='color: #FFD700; text-decoration: none;'>Shop This Look</a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("### 🎨 Dominant Colors")
                for color in analysis['dominant_colors']:
                    st.markdown(f"<div style='width:30px;height:30px;background:{color};display:inline-block;margin:5px;border-radius:4px;'></div>", unsafe_allow_html=True)

                st.markdown(f"### 🖌️ {analysis['palette_name']} Palette")
                for color in analysis['palette_colors']:
                    st.markdown(
                        f"""
                        <div style="display: flex; align-items: center; margin: 5px;">
                            <div style="width:30px;height:30px;background:{color};margin-right:10px;border-radius:4px;border: 1px solid #ddd;"></div>
                            <span style="font-size: 0.9em;">{color}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("### 💍 Accessories")
                for item in analysis['accessories']:
                    st.markdown(f"<div class='card'><p>{item}</p></div>", unsafe_allow_html=True)

                st.markdown("### 👗 Outfit Ideas (Powered by AI)")
                outfits = generate_outfit_ideas(analysis)
                for i, outfit in enumerate(outfits):
                    st.markdown(
                        f"""
                        <div class='card'>
                            <h4>Look {i+1}</h4>
                            <p>{outfit}</p>
                            <a href='#' style='color: #FFD700; text-decoration: none;'>Shop This Look</a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("### 🛍️ Store Suggestions")
                gemini_color = analysis.get("gemini_description", "").split(",")[0].split(":")[-1].strip() if analysis.get("gemini_description") else None
                stores = recommend_stores(analysis['style'], st.session_state.budget, analysis['gender'], gemini_color)
                for store in stores:
                    st.markdown(
                        f"""
                        <div class='card'>
                            <p>{store}</p>
                            <a href='#' style='color: #FFD700; text-decoration: none;'>Visit Store</a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("### 🛒 Shop These Products")
                product_links = get_product_links(analysis)
                if product_links:
                    for product in product_links:
                        st.markdown(
                            f"""
                            <div class='card'>
                                <p>{product['name']} ({product['color']}, {product['price']})</p>
                                <a href='{product['url']}' style='color: #FFD700; text-decoration: none;'>Shop Now</a>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.write("No matching products found.")

                st.markdown("### 📣 Community Feedback (Simulated)")
                st.markdown("See what the community thinks of your style! *Note: This is simulated data. For real-time X integration, check [xAI's API](https://x.ai/api).*")
                posts = simulate_x_posts(analysis['style'])
                sentiment_counts = analyze_sentiment(posts)
                for post in posts:
                    st.markdown(f"<div class='card'><p>{post}</p></div>", unsafe_allow_html=True)
                
                plt.figure(figsize=(6, 4))
                sns.barplot(x=list(sentiment_counts.keys()), y=list(sentiment_counts.values()), palette="viridis")
                plt.title("Community Sentiment", fontsize=14)
                plt.ylabel("Number of Posts", fontsize=12)
                plt.xlabel("Sentiment", fontsize=12)
                st.pyplot(plt)
                plt.clf()

                if st.button("💾 Save Recommendations"):
                    save_recommendations(analysis, outfits, stores, product_links)
                    st.success("Recommendations saved! +5 points 🌟")

    with trends_tab:
        st.subheader("👗 Trending Outfit Recommendations")
        for outfit in PRE_CURATED_OUTFITS:
            st.markdown(
                f"""
                <div class='card'>
                    <h4>{outfit['name']} ({outfit['trend']})</h4>
                    <p><strong>Description:</strong> {outfit['description']}</p>
                    <p><strong>Style Tips:</strong> {outfit['style_tips']}</p>
                    <a href='{outfit['image_url']}' style='color: #FFD700; text-decoration: none;'>View Look</a>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader("🌟 2025 Fashion Trends")
        trends = list(TREND_DATA.keys())
        scores = [TREND_DATA[trend]["score"] for trend in trends]
        plt.figure(figsize=(min(10, len(trends) * 1.5), 6))
        sns.set_style("whitegrid")
        sns.barplot(x=trends, y=scores, palette="viridis")
        plt.title("2025 Fashion Trends Prominence", fontsize=16, pad=20)
        plt.ylabel("Prominence Score (0-10)", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()

        st.markdown("### Trend Details")
        for trend in trends:
            with st.expander(f"📌 {trend} (Score: {TREND_DATA[trend]['score']}/10)"):
                st.markdown(f"**Description**: {TREND_DATA[trend]['description']}")

if __name__ == "__main__":
    main()