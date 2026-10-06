import os
import json
import base64
import httpx
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from dotenv import load_dotenv
from knowledge import SCHEMES, IPM_RULES

load_dotenv()

# ============================================================
#  MANDI CACHE — file-based, 1-hour TTL
# ============================================================
import time
from pathlib import Path

_CACHE_FILE = Path(__file__).parent / "mandi_cache.json"
_MANDI_CACHE_TTL_SECONDS = 3600   # 1 hour


def _load_mandi_cache() -> dict:
    try:
        if _CACHE_FILE.exists():
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_mandi_cache(cache: dict):
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _cache_key(state: str, district: str) -> str:
    return f"{state.strip().lower()}|{district.strip().lower()}"


def _get_cached_mandi(state: str, district: str):
    """Return cached response if fresh (< 1 hour old), else None."""
    cache = _load_mandi_cache()
    entry = cache.get(_cache_key(state, district))
    if not entry:
        return None
    age = time.time() - entry.get("ts", 0)
    if age >= _MANDI_CACHE_TTL_SECONDS:
        return None
    return entry.get("data")


def _set_cached_mandi(state: str, district: str, data: dict):
    cache = _load_mandi_cache()
    cache[_cache_key(state, district)] = {
        "ts": time.time(),
        "data": data,
    }
    _save_mandi_cache(cache)

app = FastAPI(title="Kishan Sathi API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_KEY = os.getenv("TAVILY_API_KEY", "")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY", "")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TAVILY_URL = "https://api.tavily.com/search"
# ============================================================
#  Nearest Indian city lookup — used when Nominatim fails
# ============================================================
INDIAN_CITIES = [
    ("Patna", "Bihar", 25.5941, 85.1376),
    ("Gaya", "Bihar", 24.7955, 84.9994),
    ("Muzaffarpur", "Bihar", 26.1209, 85.3647),
    ("Bhagalpur", "Bihar", 25.2425, 86.9842),
    ("Darbhanga", "Bihar", 26.1542, 85.8918),
    ("Purnia", "Bihar", 25.7771, 87.4753),
    ("Ara", "Bihar", 25.5541, 84.6605),
    ("Chapra", "Bihar", 25.7812, 84.7475),
    ("Begusarai", "Bihar", 25.4182, 86.1272),
    ("Katihar", "Bihar", 25.5541, 87.5588),
    ("Munger", "Bihar", 25.3708, 86.4734),
    ("Saharsa", "Bihar", 25.8802, 86.6000),
    ("Sasaram", "Bihar", 24.9518, 84.0313),
    ("Hajipur", "Bihar", 25.6858, 85.2094),
    ("Motihari", "Bihar", 26.6472, 84.9167),

    ("Lucknow", "Uttar Pradesh", 26.8467, 80.9462),
    ("Kanpur", "Uttar Pradesh", 26.4499, 80.3319),
    ("Varanasi", "Uttar Pradesh", 25.3176, 82.9739),
    ("Agra", "Uttar Pradesh", 27.1767, 78.0081),
    ("Prayagraj", "Uttar Pradesh", 25.4358, 81.8463),
    ("Meerut", "Uttar Pradesh", 28.9845, 77.7064),
    ("Ghaziabad", "Uttar Pradesh", 28.6692, 77.4538),
    ("Gorakhpur", "Uttar Pradesh", 26.7606, 83.3732),
    ("Bareilly", "Uttar Pradesh", 28.3670, 79.4304),
    ("Aligarh", "Uttar Pradesh", 27.8974, 78.0880),
    ("Moradabad", "Uttar Pradesh", 28.8386, 78.7733),
    ("Saharanpur", "Uttar Pradesh", 29.9640, 77.5460),
    ("Jhansi", "Uttar Pradesh", 25.4484, 78.5685),

    ("New Delhi", "Delhi", 28.6139, 77.2090),
    ("Mumbai", "Maharashtra", 19.0760, 72.8777),
    ("Pune", "Maharashtra", 18.5204, 73.8567),
    ("Nagpur", "Maharashtra", 21.1458, 79.0882),
    ("Nashik", "Maharashtra", 19.9975, 73.7898),
    ("Aurangabad", "Maharashtra", 19.8762, 75.3433),

    ("Jaipur", "Rajasthan", 26.9124, 75.7873),
    ("Jodhpur", "Rajasthan", 26.2389, 73.0243),
    ("Kota", "Rajasthan", 25.2138, 75.8648),
    ("Udaipur", "Rajasthan", 24.5854, 73.7125),

    ("Ahmedabad", "Gujarat", 23.0225, 72.5714),
    ("Surat", "Gujarat", 21.1702, 72.8311),
    ("Rajkot", "Gujarat", 22.3039, 70.8022),
    ("Vadodara", "Gujarat", 22.3072, 73.1812),

    ("Kolkata", "West Bengal", 22.5726, 88.3639),
    ("Howrah", "West Bengal", 22.5958, 88.2636),
    ("Siliguri", "West Bengal", 26.7271, 88.3953),

    ("Bhopal", "Madhya Pradesh", 23.2599, 77.4126),
    ("Indore", "Madhya Pradesh", 22.7196, 75.8577),
    ("Gwalior", "Madhya Pradesh", 26.2183, 78.1828),
    ("Jabalpur", "Madhya Pradesh", 23.1815, 79.9864),

    ("Chennai", "Tamil Nadu", 13.0827, 80.2707),
    ("Coimbatore", "Tamil Nadu", 11.0168, 76.9558),
    ("Madurai", "Tamil Nadu", 9.9252, 78.1198),

    ("Bengaluru", "Karnataka", 12.9716, 77.5946),
    ("Mysuru", "Karnataka", 12.2958, 76.6394),
    ("Hubballi", "Karnataka", 15.3647, 75.1240),

    ("Hyderabad", "Telangana", 17.3850, 78.4867),
    ("Warangal", "Telangana", 17.9689, 79.5941),

    ("Visakhapatnam", "Andhra Pradesh", 17.6868, 83.2185),
    ("Vijayawada", "Andhra Pradesh", 16.5062, 80.6480),
    ("Guntur", "Andhra Pradesh", 16.3067, 80.4365),

    ("Thiruvananthapuram", "Kerala", 8.5241, 76.9366),
    ("Kochi", "Kerala", 9.9312, 76.2673),
    ("Kozhikode", "Kerala", 11.2588, 75.7804),

    ("Bhubaneswar", "Odisha", 20.2961, 85.8245),
    ("Cuttack", "Odisha", 20.4625, 85.8830),

    ("Raipur", "Chhattisgarh", 21.2514, 81.6296),
    ("Bilaspur", "Chhattisgarh", 22.0797, 82.1409),

    ("Ranchi", "Jharkhand", 23.3441, 85.3096),
    ("Jamshedpur", "Jharkhand", 22.8046, 86.2029),
    ("Dhanbad", "Jharkhand", 23.7957, 86.4304),

    ("Guwahati", "Assam", 26.1445, 91.7362),
    ("Dibrugarh", "Assam", 27.4728, 94.9120),

    ("Chandigarh", "Punjab", 30.7333, 76.7794),
    ("Ludhiana", "Punjab", 30.9010, 75.8573),
    ("Amritsar", "Punjab", 31.6340, 74.8723),

    ("Dehradun", "Uttarakhand", 30.3165, 78.0322),
    ("Shimla", "Himachal Pradesh", 31.1048, 77.1734),
    ("Srinagar", "Jammu and Kashmir", 34.0837, 74.7973),
    ("Jammu", "Jammu and Kashmir", 32.7266, 74.8570),
]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _nearest_city(lat: float, lon: float):
    """Return (city, state, km, exact) for the nearest known Indian city."""
    best = None
    best_km = 1e9
    for city, state, clat, clon in INDIAN_CITIES:
        d = _haversine_km(lat, lon, clat, clon)
        if d < best_km:
            best_km = d
            best = (city, state)
    return best[0], best[1], best_km


def _nearby_big_mandis(state: str, district: str, lat: float = None, lon: float = None, n: int = 3):
    """Find up to n nearby big cities in the same state (for query expansion)."""
    candidates = [(c, s, clat, clon) for c, s, clat, clon in INDIAN_CITIES if s == state]
    if not candidates:
        return []
    if lat is not None and lon is not None:
        candidates.sort(key=lambda x: _haversine_km(lat, lon, x[2], x[3]))
    return [c[0] for c in candidates[:n]]
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "allam-2-7b",
]

LANG_NAMES = {
    "hi": "Hindi", "en": "English", "bn": "Bengali", "ta": "Tamil",
    "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada",
    "ml": "Malayalam", "pa": "Punjabi", "or": "Odia", "as": "Assamese",
    "ur": "Urdu", "bho": "Bhojpuri", "mai": "Maithili", "mag": "Magahi",
}


# ============================================================
#  System prompt
# ============================================================

def build_system_prompt(context: str, target_lang: str = "Hindi") -> str:
    return (
        "You are Kishan Sathi, a voice assistant for Indian farmers. "
        "You are speaking out loud to a farmer who may not read. "
        "Your reply will be read by Android Text-to-Speech.\n\n"
        "HARD RULES:\n"
        "1. PESTICIDE GUARDRAIL: Never invent a pesticide brand, dose, or chemical name.\n"
        + IPM_RULES + "\n"
        "2. Never invent weather numbers. Use ONLY numbers in CONTEXT below.\n"
        "3. Never invent mandi prices. If CONTEXT has no price, say: "
        "'I do not have today's price. Please check with your local mandi, or call 1800-180-1551.'\n"
        "4. Never invent scheme eligibility. Use only SCHEMES below.\n"
        "5. Never claim to be a doctor or vet. Livestock: 1962. Human: 108 or 112.\n"
        "6. If unsure, say so and offer a human helpline.\n"
        f"7. REPLY ONLY IN {target_lang}. Do not mix languages.\n"
        "8. Keep replies to 2-4 short sentences for spoken answers. "
        "No lists, no markdown, no emojis.\n\n"
        "SCHEMES:\n" + json.dumps(SCHEMES, ensure_ascii=False, indent=2) + "\n\n"
        "CONTEXT (live data, may be empty):\n" + context
    )


class ChatIn(BaseModel):
    question: str
    lang: str = "hi"
    lat: float | None = None
    lon: float | None = None
    district: str | None = None
    state: str | None = None


class StorageQ(BaseModel):
    crop: str
    current_price: float | None = None
    district: str | None = None
    state: str | None = None
    lang: str = "hi-IN"


# ============================================================
#  Groq chat
# ============================================================

async def groq_chat(system_prompt: str, user_text: str, max_tokens: int = 500):
    """
    Robust: tries each model, treats short/empty responses as failures,
    retries the whole chain twice before giving up.
    """
    if not GROQ_KEY:
        raise HTTPException(500, "GROQ_API_KEY missing")

    headers = {
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json",
    }

    errors = []

    # Two full passes through the model list
    for pass_num in range(2):
        for model in GROQ_MODELS:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                "temperature": 0.4,
                "max_tokens": max_tokens,
            }
            try:
                async with httpx.AsyncClient(timeout=30) as c:
                    r = await c.post(GROQ_URL, headers=headers, json=payload)

                if r.status_code != 200:
                    errors.append(f"{model}: HTTP {r.status_code}")
                    continue

                data = r.json()
                try:
                    text = (data["choices"][0]["message"]["content"] or "").strip()
                except (KeyError, IndexError):
                    errors.append(f"{model}: bad shape")
                    continue

                # Treat short/empty as failure → try next model
                if len(text) >= 20:
                    return text

                errors.append(f"{model}: too short ({len(text)} chars)")

            except Exception as e:
                errors.append(f"{model}: {str(e)[:60]}")
                continue

        # Small delay before second pass
        if pass_num == 0:
            await asyncio.sleep(1.2)

    raise HTTPException(502, "Groq failed all attempts: " + " | ".join(errors))

# ============================================================
#  Weather
# ============================================================

# ============================================================
#  WEATHER — Open-Meteo with 1-hour cache (no external key needed)
# ============================================================
_WEATHER_CACHE = {}
_WEATHER_TTL = 60 * 60   # 1 hour


@app.get("/weather")
async def weather(lat: float = 28.61, lon: float = 77.20):
    """
    Weather with 1-hour cache. Uses Open-Meteo only.
    Cache prevents rate limits. Never returns 500.
    """
    cache_key = f"{round(lat, 2)},{round(lon, 2)}"
    now = time.time()

    # 1. Cache hit → return immediately
    cached = _WEATHER_CACHE.get(cache_key)
    if cached and (now - cached["ts"]) < _WEATHER_TTL:
        out = dict(cached["data"])
        out["cached"] = True
        return out

    # 2. Fetch from Open-Meteo (single attempt, short timeout)
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m"
        "&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "precipitation_probability_max,wind_speed_10m_max,sunrise,sunset"
        "&forecast_days=7&timezone=Asia%2FKolkata"
    )

    data = None
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(url)
        if r.status_code == 200:
            data = r.json()
    except Exception:
        pass

    # 3. If fetch failed → return stale cache if available (any age)
    if data is None:
        if cached:
            out = dict(cached["data"])
            out["cached"] = True
            out["stale"] = True
            return out
        # No cache at all → graceful fallback
        return {
            "current": {
                "temp": 30.0,
                "humidity": 60,
                "rain": 0.0,
                "wind": 10.0,
                "code": 0,
                "desc": "मौसम डेटा अभी उपलब्ध नहीं",
            },
            "days": [],
            "advice": "मौसम सेवा अभी व्यस्त है। थोड़ी देर बाद कोशिश करें।",
            "fallback": True,
        }

    # 4. Parse response
    cur = data.get("current", {})
    daily = data.get("daily", {})

    days = []
    for i in range(len(daily.get("time", []))):
        days.append({
            "date": daily["time"][i],
            "code": daily["weather_code"][i],
            "desc": code_to_hi(daily["weather_code"][i]),
            "tmax": daily["temperature_2m_max"][i],
            "tmin": daily["temperature_2m_min"][i],
            "rain_mm": daily["precipitation_sum"][i],
            "rain_pct": daily["precipitation_probability_max"][i],
            "wind_kmh": daily["wind_speed_10m_max"][i],
            "sunrise": daily["sunrise"][i][-5:],
            "sunset": daily["sunset"][i][-5:],
        })

    # 5. Farming advice
    advice = ""
    rain24 = days[0]["rain_mm"] if days else 0
    tmax_today = days[0]["tmax"] if days else 0
    if rain24 > 10:
        advice = "आज तेज़ बारिश की संभावना। खेत में पानी निकासी का इंतज़ाम करें।"
    elif rain24 > 2:
        advice = "हल्की बारिश संभव। सिंचाई टाल दें।"
    elif tmax_today > 38:
        advice = "तेज़ गर्मी। सुबह या शाम को सिंचाई करें।"
    elif tmax_today < 15:
        advice = "ठंड ज़्यादा है। रात में हल्की सिंचाई करें।"
    else:
        advice = "मौसम ठीक है। सामान्य काम कर सकते हैं।"

    result = {
        "current": {
            "temp": cur.get("temperature_2m"),
            "humidity": cur.get("relative_humidity_2m"),
            "rain": cur.get("precipitation"),
            "wind": cur.get("wind_speed_10m"),
            "code": cur.get("weather_code"),
            "desc": code_to_hi(cur.get("weather_code", 0)),
        },
        "days": days,
        "advice": advice,
        "cached": False,
    }

    # 6. Store in cache
    _WEATHER_CACHE[cache_key] = {"ts": now, "data": result}

    return result
#  NEWS — Google News RSS (Hindi, real agriculture news)
# ============================================================
import xml.etree.ElementTree as _ET

_NEWS_CACHE = {"ts": 0, "data": None}
_NEWS_TTL = 1800   # 30 minutes

_WEATHER_CACHE = {}
_WEATHER_TTL = 30 * 60

@app.get("/news")
async def news():
    """
    Real Hindi agriculture news from Google News RSS.
    100% free, no API key. Cached 30 minutes.
    """
    now = time.time()
    if _NEWS_CACHE["data"] and (now - _NEWS_CACHE["ts"]) < _NEWS_TTL:
        out = dict(_NEWS_CACHE["data"])
        out["cached"] = True
        return out

    # Query in Hindi — kisan, kheti, krishi, fasal, mandi
    q = "किसान+OR+खेती+OR+कृषि+OR+फसल+OR+मंडी"
    url = (
        f"https://news.google.com/rss/search?q={q}"
        f"&hl=hi-IN&gl=IN&ceid=IN:hi"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36",
    }

    items = []
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as c:
            r = await c.get(url, headers=headers)
        if r.status_code == 200:
            root = _ET.fromstring(r.text)
            for it in root.iter("item"):
                title = (it.findtext("title") or "").strip()
                link = (it.findtext("link") or "").strip()
                pub = (it.findtext("pubDate") or "").strip()
                src_el = it.find("source")
                src = (src_el.text or "").strip() if src_el is not None else ""

                # Google News titles often end with " - Source Name"
                if src and title.endswith(f" - {src}"):
                    title = title[: -(len(src) + 3)].strip()

                if title and link:
                    items.append({
                        "title": title,
                        "detail": f"{src} • {_human_time(pub)}" if src else _human_time(pub),
                        "url": link,
                        "source": src,
                        "published": pub,
                    })
                if len(items) >= 12:
                    break
    except Exception as e:
        items = []
        _NEWS_CACHE["data"] = {"items": [], "error": f"rss_{str(e)[:50]}"}
        _NEWS_CACHE["ts"] = now
        return _NEWS_CACHE["data"]

    if not items:
        return {"items": [], "cached": False, "error": "no_news_available"}

    result = {
        "items": items,
        "cached": False,
        "updated_at": int(now),
    }
    _NEWS_CACHE["data"] = result
    _NEWS_CACHE["ts"] = now
    return result


def _human_time(rfc822: str) -> str:
    """Convert RSS pubDate to 'X घंटे पहले'."""
    if not rfc822:
        return ""
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(rfc822)
        delta = time.time() - dt.timestamp()
        if delta < 3600:
            mins = int(delta / 60)
            return f"{mins} मिनट पहले" if mins > 0 else "अभी"
        if delta < 86400:
            hrs = int(delta / 3600)
            return f"{hrs} घंटे पहले"
        days = int(delta / 86400)
        return f"{days} दिन पहले"
    except Exception:
        return ""
# ============================================================
#  Endpoints
# ============================================================

@app.get("/health")
async def health():
    return {
        "ok": True,
        "groq": bool(GROQ_KEY),
        "tavily": bool(TAVILY_KEY),
        "elevenlabs": bool(ELEVEN_KEY),
    }


@app.post("/chat")
async def chat(body: ChatIn):
    weather = await fetch_weather(body.lat, body.lon)
    bits = []
    if weather:
        c = weather.get("current", {})
        bits.append(
            f"Weather: {c.get('temperature_2m')}C, humidity {c.get('relative_humidity_2m')}%, "
            f"rain {c.get('precipitation')}mm, wind {c.get('wind_speed_10m')} km/h."
        )
    if body.district and body.state:
        bits.append(f"District: {body.district}, {body.state}.")
    context = "\n".join(bits) if bits else "(no live data)"

    lang_name = LANG_NAMES.get(body.lang.split("-")[0].lower(), "Hindi")
    reply = await groq_chat(build_system_prompt(context, lang_name), body.question, max_tokens=500)
    return {"reply": reply, "weather": weather, "lang_detected": body.lang}


@app.post("/storage-advice")
async def storage_advice(body: StorageQ):
    lang_name = LANG_NAMES.get(body.lang.split("-")[0].lower(), "Hindi")
    price_line = (
        f"Farmer's reported current price for {body.crop} is ₹{body.current_price}/quintal. "
        if body.current_price else ""
    )
    location_line = (
        f"Location: {body.district}, {body.state}. " if body.district else ""
    )
    system = (
        f"You are an agricultural market advisor for Indian farmers. Reply ONLY in {lang_name}.\n\n"
        "Give ONE clear recommendation — SELL NOW, or HOLD for X weeks. Then 2-3 short reasons.\n"
        "RULES:\n"
        "- Never invent a specific price.\n"
        "- Consider harvest season, storage lifespan, perishability, MSP timing.\n"
        "- Perishables (tomato, onion, leafy greens): usually SELL NOW.\n"
        "- Non-perishables (wheat, rice, maize, pulses): HOLD if peak harvest, SELL if off-season.\n"
        "- Write for spoken delivery: 3-5 short sentences. No lists, no emojis.\n"
        "- ALWAYS end with one short sentence in " + lang_name + " that says this is AI advice, "
        "used only for understanding the market — verify at the mandi before selling.\n"
    )
    user = (
        f"Crop: {body.crop}. {price_line}{location_line}"
        "Should I sell now or hold? Give a clear recommendation."
    )
    reply = await groq_chat(system, user, max_tokens=400)
    return {"reply": reply}


@app.post("/disease")
async def disease(image: UploadFile = File(...), lang: str = Form("hi")):
    if not GROQ_KEY:
        raise HTTPException(500, "GROQ_API_KEY missing")

    img_bytes = await image.read()
    if len(img_bytes) < 100:
        raise HTTPException(400, "Image is empty or too small")
    if len(img_bytes) > 8 * 1024 * 1024:
        raise HTTPException(400, "Image too large (max 8 MB)")

    b64 = base64.b64encode(img_bytes).decode()
    mime = image.content_type or "image/jpeg"
    if "png" in mime:
        mime = "image/png"
    elif "webp" in mime:
        mime = "image/webp"
    else:
        mime = "image/jpeg"

    lang_name = LANG_NAMES.get(lang.split("-")[0].lower(), "Hindi")

    sys_prompt = (
        f"You are an experienced Indian crop pathologist. Reply ONLY in {lang_name}.\n\n"
        "Identify the specific disease and give advice THAT SPECIFIC DISEASE needs. "
        "Do NOT give generic advice.\n\n"
        "STRUCTURE (5-7 spoken sentences):\n"
        "1. Describe the SPECIFIC visual symptoms (color, shape, location on leaf).\n"
        "2. Name the most likely disease with a confidence word (likely / possibly / not sure). "
        "Be specific — 'early blight', 'yellow rust', 'powdery mildew' — not just 'fungal disease'.\n"
        "3. Give the SPECIFIC cultural/mechanical action for THAT disease.\n"
        "4. Give the SPECIFIC biological/organic option for THAT disease.\n"
        "5. If severe or unsure, ask farmer to show leaf to KVK officer. Helpline 1800-180-1551.\n\n"
        "HARD RULES:\n"
        "- If you cannot see the disease clearly, say so honestly. Do not guess.\n"
        "- NEVER name a pesticide brand, chemical name, or dose.\n"
        "- Do NOT use the same advice for every photo.\n"
        "- No markdown, no lists, no emojis.\n"
    )

    payload = {
        "model": "qwen/qwen3.8-27b",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Diagnose the crop disease in this photo."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]},
        ],
        "temperature": 0.2,
        "max_tokens": 500,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=90) as c:
        r = await c.post(GROQ_URL, headers=headers, json=payload)

    if r.status_code != 200:
        raise HTTPException(502, f"Vision model {r.status_code}: {r.text[:400]}")

    data = r.json()
    try:
        text = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        raise HTTPException(502, f"Bad response shape: {json.dumps(data)[:300]}")

    if not text:
        text = ("फोटो साफ़ नहीं है या पत्ती पहचानी नहीं जा सकी। "
                "कृपया अच्छी रोशनी में दोबारा फोटो लें। "
                "अगर समस्या गंभीर है तो KVK हेल्पलाइन 1800-180-1551 पर कॉल करें।")

    return {"reply": text}


@app.get("/geocode")
async def geocode(lat: float, lon: float):
    """Nominatim first, nearest known Indian city as fallback."""
    # 1. Try Nominatim
    try:
        url = (f"https://nominatim.openstreetmap.org/reverse?format=json"
               f"&lat={lat}&lon={lon}&accept-language=en&zoom=10")
        async with httpx.AsyncClient(timeout=8, headers={"User-Agent": "KishanSathi/1.0"}) as c:
            j = (await c.get(url)).json()
        a = j.get("address", {})
        st = a.get("state")
        dist = a.get("state_district") or a.get("county") or a.get("city")
        vil = a.get("village") or a.get("town") or a.get("suburb")

        if st and dist:
            return {
                "state": st,
                "district": dist,
                "village": vil,
                "source": "nominatim",
                "nearest_city": None,
                "distance_km": None,
            }
    except Exception:
        pass

    # 2. Fallback — nearest known city
    city, state, km = _nearest_city(lat, lon)
    return {
        "state": state,
        "district": city,
        "village": None,
        "source": "nearest_city",
        "nearest_city": city,
        "distance_km": round(km, 1),
    }


@app.get("/weather")
async def weather(lat: float = 28.61, lon: float = 77.20):
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m"
        "&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "precipitation_probability_max,wind_speed_10m_max,sunrise,sunset"
        "&forecast_days=7&timezone=Asia%2FKolkata"
    )
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(url)
        if r.status_code != 200:
            raise HTTPException(502, f"Open-Meteo: {r.text[:200]}")
        data = r.json()

    cur = data.get("current", {})
    daily = data.get("daily", {})

    days = []
    for i in range(len(daily.get("time", []))):
        days.append({
            "date": daily["time"][i],
            "code": daily["weather_code"][i],
            "desc": code_to_hi(daily["weather_code"][i]),
            "tmax": daily["temperature_2m_max"][i],
            "tmin": daily["temperature_2m_min"][i],
            "rain_mm": daily["precipitation_sum"][i],
            "rain_pct": daily["precipitation_probability_max"][i],
            "wind_kmh": daily["wind_speed_10m_max"][i],
            "sunrise": daily["sunrise"][i][-5:],
            "sunset": daily["sunset"][i][-5:],
        })

    advice = ""
    rain24 = days[0]["rain_mm"] if days else 0
    tmax_today = days[0]["tmax"] if days else 0
    if rain24 > 10:
        advice = "आज तेज़ बारिश की संभावना। खेत में पानी निकासी का इंतज़ाम करें।"
    elif rain24 > 2:
        advice = "हल्की बारिश संभव। सिंचाई टाल दें।"
    elif tmax_today > 38:
        advice = "तेज़ गर्मी। सुबह या शाम को सिंचाई करें।"
    elif tmax_today < 15:
        advice = "ठंड ज़्यादा है। रात में हल्की सिंचाई करें।"
    else:
        advice = "मौसम ठीक है। सामान्य काम कर सकते हैं।"

    return {
        "current": {
            "temp": cur.get("temperature_2m"),
            "humidity": cur.get("relative_humidity_2m"),
            "rain": cur.get("precipitation"),
            "wind": cur.get("wind_speed_10m"),
            "code": cur.get("weather_code"),
            "desc": code_to_hi(cur.get("weather_code", 0)),
        },
        "days": days,
        "advice": advice,
    }


# ============================================================
#  MANDI — Hybrid (Tavily search + fallback)
# ============================================================

def _sample_mandi(state: str, district: str):
    """Fallback sample data — used only if live sources fail."""
    sample = [
        {"crop": "गेहूं", "variety": "सामान्य", "min": 2150, "max": 2450, "modal": 2320, "trend": "up",   "change": 120},
        {"crop": "धान",   "variety": "सामान्य", "min": 2050, "max": 2350, "modal": 2200, "trend": "up",   "change": 80},
        {"crop": "मक्का", "variety": "सामान्य", "min": 1850, "max": 2100, "modal": 1960, "trend": "down", "change": -45},
        {"crop": "सरसों", "variety": "सामान्य", "min": 5100, "max": 5600, "modal": 5350, "trend": "up",   "change": 200},
        {"crop": "टमाटर", "variety": "सामान्य", "min": 800,  "max": 1400, "modal": 1100, "trend": "down", "change": -150},
        {"crop": "आलू",   "variety": "सामान्य", "min": 950,  "max": 1250, "modal": 1080, "trend": "up",   "change": 60},
        {"crop": "प्याज", "variety": "सामान्य", "min": 1450, "max": 1800, "modal": 1650, "trend": "up",   "change": 90},
        {"crop": "गन्ना", "variety": "सामान्य", "min": 320,  "max": 380,  "modal": 350,  "trend": "flat", "change": 0},
    ]
    return {
        "state": state,
        "district": district,
        "updated": "आज (sample)",
        "source": "sample data — live not available",
        "live": False,
        "items": sample,
    }


CROP_EN_MAP = {
    "गेहूं": "Wheat", "गेहू": "Wheat", "wheat": "Wheat",
    "धान": "Paddy(Dhan)", "चावल": "Rice", "paddy": "Paddy(Dhan)",
    "मक्का": "Maize", "maize": "Maize", "corn": "Maize",
    "सरसों": "Mustard", "mustard": "Mustard",
    "टमाटर": "Tomato", "tomato": "Tomato",
    "आलू": "Potato", "potato": "Potato",
    "प्याज": "Onion", "onion": "Onion",
    "गन्ना": "Sugarcane", "sugarcane": "Sugarcane",
    "चना": "Bengal Gram(Gram)(Whole)", "chana": "Bengal Gram(Gram)(Whole)",
    "मूंग": "Green Gram (Moong)(Whole)", "moong": "Green Gram (Moong)(Whole)",
    "सोयाबीन": "Soyabean", "soyabean": "Soyabean",
    "बाजरा": "Bajra(Pearl Millet/Cumbu)", "bajra": "Bajra(Pearl Millet/Cumbu)",
    "ज्वार": "Jowar(Sorghum)", "jowar": "Jowar(Sorghum)",
    "मूंगफली": "Groundnut", "groundnut": "Groundnut",
}


async def _tavily_search_mandi(crop_en: str, district: str, state: str):
    """Search Tavily for live mandi prices. Returns combined snippets or None."""
    if not TAVILY_KEY:
        return None
    query = (
        f"{crop_en} mandi price today in {district} {state} India "
        f"per quintal modal price"
    )
    payload = {
        "query": query,
        "max_results": 6,
        "search_depth": "advanced",
        "include_domains": [
            "agmarknet.gov.in",
            "commodityonline.com",
            "agriwatch.com",
            "commoditiescontrol.com",
            "mandi.farm",
            "krishimarket.com",
            "indiaagristat.com",
        ],
    }
    headers = {
        "Authorization": f"Bearer {TAVILY_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=25) as c:
            r = await c.post(TAVILY_URL, headers=headers, json=payload)
        if r.status_code != 200:
            return None
        data = r.json()
        results = data.get("results", [])
        if not results:
            return None
        combined = "\n\n".join(
            f"Source: {res.get('title','')} ({res.get('url','')})\n{res.get('content','')}"
            for res in results[:6]
        )
        return combined
    except Exception:
        return None


async def _extract_prices_from_snippets(snippets: str, crop_en: str, district: str, state: str):
    """Ask Groq to extract structured prices from search snippets."""
    if not GROQ_KEY:
        return None
    system = (
        "You extract structured mandi (Indian agricultural market) price data from search snippets. "
        "Return ONLY valid JSON. No prose, no markdown.\n\n"
        "Output format:\n"
        "{\n"
        '  "found": true,\n'
        '  "items": [\n'
        '    {"crop": "...", "variety": "सामान्य", "min": 2000, "max": 2400, "modal": 2200, "market": "Patna"}\n'
        "  ],\n"
        '  "confidence": "high|medium|low"\n'
        "}\n\n"
        "Rules:\n"
        "- Only include items where a clear ₹ price per quintal is visible in the snippets.\n"
        "- If no clear price is found, return: {\"found\": false, \"items\": [], \"confidence\": \"low\"}\n"
        "- Do NOT invent prices. If unsure, return found: false.\n"
        "- All prices must be integers (₹ per quintal).\n"
    )
    user = (
        f"Crop: {crop_en}\nDistrict: {district}, {state}\n\n"
        f"SNIPPETS:\n{snippets}\n\n"
        "Extract up to 5 price items as JSON."
    )
    try:
        text = await groq_chat(system, user, max_tokens=500)
    except Exception:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip("` \n")
    try:
        return json.loads(text)
    except Exception:
        return None



@app.get("/mandi-list")
async def mandi_list(
    state: str = "Bihar",
    district: str = "Patna",
    lat: float | None = None,
    lon: float | None = None,
    force: bool = False,
):
    """
    1-hour cache per (state, district).
    First call fetches live + sample. Subsequent calls return cached for 1 hour.
    Pass force=true to bypass cache (for testing).
    """
    # 1. Check cache
    if not force:
        cached = _get_cached_mandi(state, district)
        if cached is not None:
            cached["cached"] = True
            return cached

    # 2. Build fresh response
    base = _sample_mandi(state, district)
    items = base["items"]
    any_live = False

    # 3. Find nearby big mandis for search expansion
    nearby = _nearby_big_mandis(state, district, lat, lon, n=3)
    loc_terms = [district] + [c for c in nearby if c.lower() != district.lower()]

    # 4. Try live data for the main crops
    live_crops = {
        "गेहूं": "Wheat",
        "धान": "Paddy(Dhan)",
        "आलू": "Potato",
        "प्याज": "Onion",
    }

    for crop_hi, crop_en in live_crops.items():
        try:
            snippets = await _tavily_search_mandi_multi(crop_en, loc_terms, state)
            if not snippets:
                continue
            extracted = await _extract_prices_from_snippets(snippets, crop_en, district, state)
            if not extracted or not extracted.get("found"):
                continue
            live_items = extracted.get("items", [])
            if not live_items:
                continue

            it = live_items[0]
            modal = int(it.get("modal", 0) or 0)
            mn = int(it.get("min", 0) or 0)
            mx = int(it.get("max", 0) or 0)
            if modal <= 0:
                continue

            for i, existing in enumerate(items):
                if existing["crop"] == crop_hi:
                    items[i] = {
                        "crop": crop_hi,
                        "variety": "सामान्य",
                        "min": mn or modal,
                        "max": mx or modal,
                        "modal": modal,
                        "trend": "flat",
                        "change": 0,
                    }
                    any_live = True
                    break
        except Exception:
            continue

    if any_live:
        base["updated"] = "आज (कुछ भाव live)"
        base["source"] = f"live web search + अनुमानित — {district}"
        base["live"] = True
    else:
        base["updated"] = "आज (अनुमानित)"
        base["source"] = "sample data — live not available"
        base["live"] = False

    # 5. Store in cache
    base["cached"] = False
    base["cached_at"] = int(time.time())
    _set_cached_mandi(state, district, base)

    return base


@app.post("/mandi-clear-cache")
async def mandi_clear_cache():
    """Clear mandi cache — for testing."""
    try:
        if _CACHE_FILE.exists():
            _CACHE_FILE.unlink()
        return {"ok": True, "msg": "cache cleared"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _tavily_search_mandi_multi(crop_en: str, loc_terms: list, state: str):
    """Search across multiple location terms, return combined snippets."""
    if not TAVILY_KEY:
        return None
    loc_phrase = " OR ".join(loc_terms[:3])
    query = f"{crop_en} mandi price today in {loc_phrase} {state} India per quintal"
    payload = {
        "query": query,
        "max_results": 8,
        "search_depth": "advanced",
        "include_domains": [
            "agmarknet.gov.in",
            "commodityonline.com",
            "agriwatch.com",
            "commoditiescontrol.com",
            "mandi.farm",
            "krishimarket.com",
            "indiaagristat.com",
        ],
    }
    headers = {
        "Authorization": f"Bearer {TAVILY_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=25) as c:
            r = await c.post(TAVILY_URL, headers=headers, json=payload)
        if r.status_code != 200:
            return None
        results = r.json().get("results", [])
        if not results:
            return None
        return "\n\n".join(
            f"Source: {res.get('title','')} ({res.get('url','')})\n{res.get('content','')}"
            for res in results[:8]
        )
    except Exception:
        return None

 # ============================================================
#  FARMER SCHEMES — with eligibility rules + status
# ============================================================

SCHEMES_DB = [
    {
        "id": "pm-kisan",
        "name_hi": "प्रधानमंत्री किसान सम्मान निधि",
        "name_en": "PM-KISAN",
        "emoji": "💰",
        "category": "आय सहायता",
        "benefit": "हर साल ₹6,000 — तीन किस्तों में ₹2,000-₹2,000।",
        "status": "OPEN",
        "status_text": "हर समय खुली — बस रजिस्ट्रेशन करें",
        "apply_steps": [
            "pmkisan.gov.in पर जाएं",
            "नया रजिस्ट्रेशन चुनें",
            "आधार, बैंक खाता, जमीन के कागज जमा करें",
            "नज़दीकी CSC सेंटर भी जा सकते हैं (₹15-20 शुल्क)",
        ],
        "helpline": "155261 / 011-24300606",
        "url": "https://pmkisan.gov.in",
        "eligibility": {
            "min_land_acres": 0.01,
            "must_be_farmer": True,
            "block_if_govt_employee": True,
            "block_if_tax_payer": True,
            "requires_aadhaar": True,
            "requires_bank_account": True,
        },
    },
    {
        "id": "pmfby",
        "name_hi": "प्रधानमंत्री फसल बीमा योजना",
        "name_en": "PMFBY",
        "emoji": "🛡️",
        "category": "फसल बीमा",
        "benefit": "प्राकृतिक आपदा, कीट या रोग से फसल खराब होने पर बीमा। प्रीमियम सिर्फ 2% (खरीफ), 1.5% (रबी)।",
        "status": "SEASONAL",
        "status_text": "खरीफ: जुलाई-अगस्त · रबी: नवंबर-दिसंबर",
        "apply_steps": [
            "pmfby.gov.in पर जाएं या बैंक जाएं",
            "बुआई प्रमाणपत्र, बैंक पासबुक, आधार ले जाएं",
            "किसान की जमीन का रिकॉर्ड",
            "बुआई के 10 दिन के अंदर-अंदर रजिस्टर करें",
        ],
        "helpline": "14447",
        "url": "https://pmfby.gov.in",
        "eligibility": {
            "min_land_acres": 0.01,
            "must_be_farmer": True,
            "requires_aadhaar": True,
            "requires_bank_account": True,
        },
    },
    {
        "id": "kcc",
        "name_hi": "किसान क्रेडिट कार्ड",
        "name_en": "Kisan Credit Card",
        "emoji": "💳",
        "category": "ऋण",
        "benefit": "₹3 लाख तक का कृषि ऋण — 4% प्रभावी ब्याज (समय पर चुकाने पर)।",
        "status": "OPEN",
        "status_text": "हर समय खुली — बैंक जाएं",
        "apply_steps": [
            "अपने बैंक (SBI, PNB, ग्रामीण बैंक) में जाएं",
            "KCC फॉर्म भरें",
            "आधार, PAN, जमीन के कागज, फोटो लगाएं",
            "7-15 दिन में कार्ड मिल जाता है",
        ],
        "helpline": "अपने बैंक की हेल्पलाइन",
        "url": "https://www.nabard.org",
        "eligibility": {
            "min_land_acres": 0.01,
            "min_age": 18,
            "requires_aadhaar": True,
            "requires_bank_account": True,
        },
    },
    {
        "id": "pm-kisan-maan-dhan",
        "name_hi": "प्रधानमंत्री किसान मानधन योजना",
        "name_en": "PM Kisan Maan Dhan Yojana",
        "emoji": "👴",
        "category": "पेंशन",
        "benefit": "60 साल के बाद हर महीने ₹3,000 पेंशन।",
        "status": "OPEN",
        "status_text": "हर समय खुली — CSC सेंटर जाएं",
        "apply_steps": [
            "नज़दीकी CSC सेंटर जाएं",
            "आधार, बैंक पासबुक, मोबाइल नंबर ले जाएं",
            "₹55 से ₹200 तक की मासिक किस्त चुनें",
            "हर महीने की किस्त अपने बैंक से कटेगी",
        ],
        "helpline": "1800-267-6888",
        "url": "https://maandhan.in",
        "eligibility": {
            "min_age": 18,
            "max_age": 40,
            "max_land_acres": 5.0,
            "must_be_farmer": True,
            "requires_aadhaar": True,
            "requires_bank_account": True,
        },
    },
    {
        "id": "soil-health",
        "name_hi": "मृदा स्वास्थ्य कार्ड",
        "name_en": "Soil Health Card",
        "emoji": "🌱",
        "category": "मिट्टी",
        "benefit": "हर 2 साल में मुफ़्त मिट्टी जाँच + खाद की सलाह।",
        "status": "OPEN",
        "status_text": "हर समय खुली",
        "apply_steps": [
            "soilhealth.dac.gov.in पर जाएं",
            "या अपने ब्लॉक कृषि कार्यालय में जाएं",
            "खेत से मिट्टी का नमूना दें",
            "कुछ हफ़्तों में कार्ड मिलता है",
        ],
        "helpline": "1800-180-1551",
        "url": "https://soilhealth.dac.gov.in",
        "eligibility": {
            "min_land_acres": 0.01,
            "must_be_farmer": True,
        },
    },
    {
        "id": "kisan-credit-subsidy",
        "name_hi": "कृषि यंत्र अनुदान योजना",
        "name_en": "Agriculture Machinery Subsidy",
        "emoji": "🚜",
        "category": "यंत्र सब्सिडी",
        "benefit": "ट्रैक्टर, रोटावेटर, पंप सेट पर 40-50% सरकारी अनुदान।",
        "status": "SEASONAL",
        "status_text": "राज्य के अनुसार खुलती है — ज़्यादातर मार्च-मई",
        "apply_steps": [
            "अपने राज्य के कृषि पोर्टल पर जाएं",
            "जैसे: Bihar → dbtagriculture.bihar.gov.in",
            "यंत्र चुनें, कोटेशन अपलोड करें",
            "स्वीकृति के बाद DBT से पैसे आते हैं",
        ],
        "helpline": "अपने जिला कृषि कार्यालय",
        "url": "https://agrimachinery.nic.in",
        "eligibility": {
            "min_land_acres": 0.5,
            "must_be_farmer": True,
            "requires_aadhaar": True,
            "requires_bank_account": True,
        },
    },
    {
        "id": "sc-st-farmer",
        "name_hi": "SC/ST किसान उपयोजना",
        "name_en": "SC/ST Farmer Support",
        "emoji": "🤝",
        "category": "विशेष वर्ग",
        "benefit": "SC/ST किसानों के लिए अतिरिक्त सब्सिडी, बीज और ट्रेनिंग।",
        "status": "OPEN",
        "status_text": "हर समय खुली (SC/ST के लिए)",
        "apply_steps": [
            "जाति प्रमाणपत्र लें (अगर नहीं है)",
            "अपने ब्लॉक कृषि कार्यालय जाएं",
            "जाति प्रमाणपत्र + आधार + बैंक पासबुक दें",
            "स्थानीय अधिकारी से सलाह लें",
        ],
        "helpline": "1800-180-1551",
        "url": "https://agricoop.gov.in",
        "eligibility": {
            "min_land_acres": 0.01,
            "must_be_farmer": True,
            "allowed_categories": ["SC", "ST"],
        },
    },
    {
        "id": "kisan-tractor-subsidy",
        "name_hi": "किसान कॉल सेंटर",
        "name_en": "Kisan Call Centre",
        "emoji": "☎️",
        "category": "सलाह",
        "benefit": "कृषि से जुड़ी हर समस्या पर मुफ़्त सलाह, 24x7 हिंदी और 21 भाषाओं में।",
        "status": "OPEN",
        "status_text": "24x7 खुला",
        "apply_steps": [
            "1800-180-1551 पर कॉल करें",
            "अपनी समस्या हिंदी में बताएं",
            "कृषि वैज्ञानिक सीधे सलाह देंगे",
        ],
        "helpline": "1800-180-1551",
        "url": "https://agricoop.gov.in",
        "eligibility": {
            "min_land_acres": 0.0,
            "must_be_farmer": True,
        },
    },
]


def _evaluate_eligibility(scheme: dict, profile: dict) -> dict:
    """
    Returns: {result: "eligible"|"maybe"|"not", reasons: [...], missing: [...]}
    """
    e = scheme.get("eligibility", {})
    reasons = []
    missing = []

    # Age
    age = profile.get("age")
    if age is not None:
        if "min_age" in e and age < e["min_age"]:
            return {"result": "not", "reasons": [f"उम्र कम से कम {e['min_age']} साल चाहिए"], "missing": []}
        if "max_age" in e and age > e["max_age"]:
            return {"result": "not", "reasons": [f"उम्र {e['max_age']} साल से ज़्यादा नहीं होनी चाहिए"], "missing": []}
        reasons.append(f"✅ उम्र {age} साल सही है")

    # Land
    land = profile.get("land_acres")
    if land is not None:
        if "min_land_acres" in e and land < e["min_land_acres"]:
            return {"result": "not", "reasons": [f"कम से कम {e['min_land_acres']} एकड़ जमीन चाहिए"], "missing": []}
        if "max_land_acres" in e and land > e["max_land_acres"]:
            return {"result": "not", "reasons": [f"सिर्फ {e['max_land_acres']} एकड़ तक वाले किसान eligible"], "missing": []}
        if "min_land_acres" in e or "max_land_acres" in e:
            reasons.append(f"✅ {land} एकड़ जमीन के साथ योग्य")

    # Govt employee
    if e.get("block_if_govt_employee") and profile.get("is_govt_employee"):
        return {"result": "not", "reasons": ["सरकारी कर्मचारी eligible नहीं हैं"], "missing": []}

    # Income tax payer
    if e.get("block_if_tax_payer") and profile.get("is_tax_payer"):
        return {"result": "not", "reasons": ["आयकर दाता eligible नहीं हैं"], "missing": []}

    # Category
    allowed = e.get("allowed_categories")
    if allowed:
        cat = profile.get("category")
        if cat and cat not in allowed:
            return {"result": "not", "reasons": [f"सिर्फ {', '.join(allowed)} वर्ग के लिए"], "missing": []}
        elif not cat:
            missing.append("जाति श्रेणी")
        else:
            reasons.append(f"✅ {cat} वर्ग के लिए योग्य")

    # Aadhaar
    if e.get("requires_aadhaar"):
        if profile.get("has_aadhaar") is False:
            return {"result": "not", "reasons": ["आधार कार्ड ज़रूरी है"], "missing": ["आधार कार्ड"]}
        elif profile.get("has_aadhaar") is None:
            missing.append("आधार कार्ड")

    # Bank account
    if e.get("requires_bank_account"):
        if profile.get("has_bank_account") is False:
            return {"result": "not", "reasons": ["बैंक खाता ज़रूरी है"], "missing": ["बैंक खाता"]}
        elif profile.get("has_bank_account") is None:
            missing.append("बैंक खाता")

    if missing:
        return {"result": "maybe", "reasons": reasons, "missing": missing}
    return {"result": "eligible", "reasons": reasons, "missing": []}


class ProfileIn(BaseModel):
    age: int | None = None
    land_acres: float | None = None
    annual_income: int | None = None
    category: str | None = None   # General/OBC/SC/ST
    state: str | None = None
    is_govt_employee: bool = False
    is_tax_payer: bool = False
    has_aadhaar: bool | None = None
    has_bank_account: bool | None = None


@app.get("/schemes")
async def schemes():
    """All schemes without eligibility evaluation."""
    return {"schemes": SCHEMES_DB}


@app.post("/schemes-match")
async def schemes_match(profile: ProfileIn):
    """
    Match all schemes against a farmer profile.
    Returns: eligible_count, maybe_count, total_open, and full list with result status.
    """
    p = profile.dict()
    results = []
    eligible = 0
    maybe = 0

    for s in SCHEMES_DB:
        ev = _evaluate_eligibility(s, p)
        entry = dict(s)
        entry["result"] = ev["result"]
        entry["reasons"] = ev["reasons"]
        entry["missing"] = ev["missing"]
        results.append(entry)

        if ev["result"] == "eligible":
            eligible += 1
        elif ev["result"] == "maybe":
            maybe += 1

    # Sort: eligible first, then maybe, then not
    order = {"eligible": 0, "maybe": 1, "not": 2}
    results.sort(key=lambda x: order[x["result"]])

    open_count = sum(1 for s in SCHEMES_DB if s["status"] in ("OPEN", "SEASONAL"))

    return {
        "eligible_count": eligible,
        "maybe_count": maybe,
        "open_count": open_count,
        "total": len(SCHEMES_DB),
        "schemes": results,
    }


# ============================================================
#  KNOWLEDGE — Live tips (Tavily) + Static categories
# ============================================================
_KNOWLEDGE_CACHE = {"ts": 0, "tips": None}
_KNOWLEDGE_TTL = 2 * 3600   # 2 hours


async def _fetch_live_tips():
    """Fetch 3 fresh agriculture tips from Tavily + Groq."""
    if not TAVILY_KEY:
        return None

    payload = {
        "query": "India agriculture advisory today crop tips farmer",
        "max_results": 8,
        "search_depth": "basic",
        "include_domains": [
            "agricoop.gov.in",
            "icar.org.in",
            "krishijagran.com",
            "agrifarming.in",
            "down-to-earth.org",
            "gaonconnection.com",
        ],
    }
    headers = {
        "Authorization": f"Bearer {TAVILY_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(TAVILY_URL, headers=headers, json=payload)
        if r.status_code != 200:
            return None
        results = r.json().get("results", [])
    except Exception:
        return None

    if not results:
        return None

    snippets = "\n\n".join(
        f"{res.get('title','')}: {res.get('content','')[:400]}"
        for res in results[:6]
    )

    system = (
        "Extract 3 short actionable agriculture tips for Indian farmers. "
        "Reply ONLY as a JSON array, no prose, no markdown:\n"
        '[{"title":"छोटा शीर्षक","detail":"2 वाक्य की सलाह"}]\n\n'
        "Rules:\n"
        "- Both title and detail must be in Hindi (Devanagari).\n"
        "- Title: 8 words max. Detail: 25 words max.\n"
        "- Each tip must be specific and actionable.\n"
        "- If no clear tips in snippets, return []\n"
    )
    user = f"Extract 3 tips:\n\n{snippets}"

    try:
        text = await groq_chat(system, user, max_tokens=700)
        text = text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:]
        text = text.strip("` \n")

        first = text.find("[")
        last = text.rfind("]")
        if first >= 0 and last > first:
            text = text[first:last + 1]

        tips = json.loads(text)
        if not isinstance(tips, list):
            return None

        clean = []
        for t in tips[:3]:
            if isinstance(t, dict) and t.get("title") and t.get("detail"):
                clean.append({
                    "title": str(t["title"])[:80],
                    "detail": str(t["detail"])[:250],
                })
        return clean if clean else None
    except Exception:
        return None


_FALLBACK_TIPS = [
    {
        "title": "सुबह की सिंचाई सबसे अच्छी",
        "detail": "सुबह 6-9 बजे सिंचाई करने से 30% पानी बचता है और फसल जल्दी बढ़ती है।"
    },
    {
        "title": "मिट्टी की जाँच ज़रूर करवाएं",
        "detail": "हर 2 साल में मुफ़्त मिट्टी जाँच करवाएं — soilhealth.dac.gov.in पर।"
    },
    {
        "title": "नीम तेल से कीट नियंत्रण",
        "detail": "3 मिली नीम तेल प्रति लीटर पानी में मिलाकर शाम को छिड़कें।"
    },
]


@app.get("/knowledge")
async def knowledge(force: bool = False):
    now = time.time()
    tips = None

    if (not force
        and _KNOWLEDGE_CACHE["tips"]
        and (now - _KNOWLEDGE_CACHE["ts"]) < _KNOWLEDGE_TTL):
        tips = _KNOWLEDGE_CACHE["tips"]

    if not tips:
        try:
            tips = await _fetch_live_tips()
        except Exception:
            tips = None
        if tips and len(tips) > 0:
            _KNOWLEDGE_CACHE["tips"] = tips
            _KNOWLEDGE_CACHE["ts"] = now

    if not tips or len(tips) == 0:
        tips = _FALLBACK_TIPS

    return {
        "live_tips": tips,
        "live_updated_at": int(_KNOWLEDGE_CACHE["ts"]) if _KNOWLEDGE_CACHE["ts"] else 0,
        "is_live": bool(_KNOWLEDGE_CACHE["ts"] and
                        (now - _KNOWLEDGE_CACHE["ts"]) < _KNOWLEDGE_TTL),
        "categories": [
            {"id": "seasons", "emoji": "🌤️", "title": "फसल के मौसम", "items": [
                {"name": "खरीफ (जून–अक्टूबर)", "detail": "धान, मक्का, ज्वार, बाजरा, कपास, सोयाबीन।"},
                {"name": "रबी (नवंबर–मार्च)", "detail": "गेहूं, जौ, चना, मटर, सरसों, मसूर। सिंचाई ज़रूरी।"},
                {"name": "ज़ायद (अप्रैल–जून)", "detail": "तरबूज़, खरबूज़, ककड़ी, खीरा, मूंग।"},
            ]},
            {"id": "soil", "emoji": "🪨", "title": "मिट्टी के प्रकार", "items": [
                {"name": "जलोढ़ मिट्टी", "detail": "गेहूं, धान, गन्ना के लिए सबसे अच्छी।"},
                {"name": "काली मिट्टी", "detail": "कपास के लिए प्रसिद्ध। नमी रोकती है।"},
                {"name": "लाल मिट्टी", "detail": "मूंगफली, मक्का अच्छे होते हैं।"},
                {"name": "रेतीली मिट्टी", "detail": "बाजरा, ज्वार, मूंगफली उगती हैं।"},
            ]},
            {"id": "fertilizer", "emoji": "🌿", "title": "खाद और उर्वरक", "items": [
                {"name": "गोबर की खाद", "detail": "हर फसल के लिए सुरक्षित। 10–15 टन/एकड़।"},
                {"name": "यूरिया (N)", "detail": "बुआई के समय आधी, फिर दो बार टॉप ड्रेसिंग।"},
                {"name": "DAP (N+P)", "detail": "जड़ मज़बूत करता है। 50 किलो/एकड़।"},
                {"name": "MOP / पोटाश (K)", "detail": "आलू, गन्ना, केले के लिए ज़रूरी।"},
                {"name": "जैविक खाद", "detail": "Trichoderma, Rhizobium — रोग कम करते हैं।"},
            ]},
            {"id": "pests", "emoji": "🐛", "title": "आम कीट", "items": [
                {"name": "तना छेदक", "detail": "फेरोमोन ट्रैप लगाएं। प्रभावित तने जला दें।"},
                {"name": "माहू", "detail": "नीम तेल का छिड़काव।"},
                {"name": "सफेद मक्खी", "detail": "पीला चिपचिपा ट्रैप लगाएं।"},
                {"name": "तना मक्खी", "detail": "बीजोपचार करें।"},
            ]},
            {"id": "organic", "emoji": "🍃", "title": "जैविक खेती", "items": [
                {"name": "जीवामृत", "detail": "10 किलो गोबर + 10 लीटर गोमूत्र + 2 किलो गुड़ + 2 किलो बेसन + 1 किलो मिट्टी।"},
                {"name": "नीम तेल", "detail": "3 मिली/लीटर पानी में। सुबह-शाम छिड़काव।"},
                {"name": "बीजोपचार", "detail": "5 ग्राम Trichoderma/किलो बीज।"},
                {"name": "पंचगव्य", "detail": "21 दिन फर्मेंट। मिट्टी की उर्वरता बढ़ाता है।"},
            ]},
            {"id": "helplines", "emoji": "☎️", "title": "ज़रूरी हेल्पलाइन", "items": [
                {"name": "किसान कॉल सेंटर", "detail": "1800-180-1551 (मुफ्त, 24x7)"},
                {"name": "कृषि विज्ञान केंद्र (KVK)", "detail": "जिले में KVK पर जाएं।"},
                {"name": "पशु चिकित्सा", "detail": "1962"},
                {"name": "आपातकाल", "detail": "112"},
                {"name": "साइबर अपराध", "detail": "1930"},
                {"name": "महिला हेल्पलाइन", "detail": "181"},
                {"name": "बाल हेल्पलाइन", "detail": "1098"},
            ]},
            {"id": "savings", "emoji": "💧", "title": "पानी की बचत", "items": [
                {"name": "ड्रिप सिंचाई", "detail": "40–60% पानी की बचत। सब्सिडी मिलती है।"},
                {"name": "स्प्रिंकलर", "detail": "गेहूं, चना, सब्जियों के लिए। 30% पानी बचता है।"},
                {"name": "मल्चिंग", "detail": "भूसे की 5 सेमी परत।"},
                {"name": "वर्षा जल संचयन", "detail": "खेत के किनारे तालाब बनाएं।"},
            ]},
        ]
    }

# ============================================================
#  STT / TTS (optional — kept for future use)
# ============================================================

@app.post("/stt")
async def stt(audio: UploadFile = File(...), lang: str = Form("")):
    if not GROQ_KEY:
        raise HTTPException(500, "GROQ_API_KEY missing")
    audio_bytes = await audio.read()
    files = {
        "file": (audio.filename or "audio.m4a", audio_bytes, audio.content_type or "audio/m4a"),
    }
    data = {"model": "whisper-large-v3", "response_format": "verbose_json", "temperature": "0"}
    if lang and len(lang) == 2:
        data["language"] = lang
    headers = {"Authorization": f"Bearer {GROQ_KEY}"}
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers=headers, files=files, data=data,
        )
    if r.status_code != 200:
        raise HTTPException(502, f"Whisper {r.status_code}: {r.text[:200]}")
    j = r.json()
    return {"text": (j.get("text") or "").strip(), "lang": lang or j.get("language") or "unknown"}


class TTSIn(BaseModel):
    text: str
    lang: str = "hi"


@app.post("/tts")
async def tts(body: TTSIn):
    if not ELEVEN_KEY or not ELEVEN_VOICE:
        raise HTTPException(500, "ElevenLabs not configured")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}"
    headers = {
        "xi-api-key": ELEVEN_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": body.text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(url, headers=headers, json=payload)
    if r.status_code != 200:
        raise HTTPException(502, f"ElevenLabs {r.status_code}: {r.text[:200]}")
    return Response(content=r.content, media_type="audio/mpeg")
