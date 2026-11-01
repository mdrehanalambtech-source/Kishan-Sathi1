@app.get("/mandi-list")
async def mandi_list(
    state: str = "Bihar",
    district: str = "Patna",
    lat: float | None = None,
    lon: float | None = None,
):
    if not district and lat and lon:
        try:
            geo = await geocode(lat=lat, lon=lon)
            district = geo.get("district") or "Patna"
            state = geo.get("state") or "Bihar"
        except Exception:
            pass

    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    
    try:
        # 1. जिले का डेटा मांगो
        params = {"api-key": DATA_GOV_KEY, "format": "json", "limit": 15, "filters[state]": state, "filters[district]": district}
        async with httpx.AsyncClient(timeout=15) as client:
            res = await client.get(url, params=params)
            records = res.json().get("records", []) if res.status_code == 200 else []
            
            # 2. अगर जिले में डेटा नहीं है, तो पूरे राज्य का डेटा लाओ
            if not records:
                params.pop("filters[district]")
                res2 = await client.get(url, params=params)
                records = res2.json().get("records", []) if res2.status_code == 200 else []
                if records:
                    district = "राज्य भर से (State)"

            if records:
                items = []
                for rec in records:
                    items.append({
                        "crop": rec.get("commodity", "फसल"),
                        "variety": rec.get("variety", "सामान्य"),
                        "min": int(float(rec.get("min_price", 0))),
                        "max": int(float(rec.get("max_price", 0))),
                        "modal": int(float(rec.get("modal_price", 0))),
                        "trend": "flat",
                        "change": 0,
                    })
                return {
                    "state": state, "district": district, "updated": records[0].get("arrival_date", "आज"),
                    "source": "Government Agmarknet (Live)", "live": True, "items": items,
                }
    except Exception as e:
        print(f"Agmarknet API Error: {e}")

    # 3. अगर सर्वर पूरी तरह डाउन है, तो सैंपल डेटा
    return _sample_mandi(state, district)


@app.post("/storage-advice")
async def storage_advice(body: StorageQ):
    district = body.district or "Patna"
    state = body.state or "Bihar"
    crop_hi = body.crop
    user_price = body.current_price or 0

    crop_map = {
        "गेहूं": "Wheat", "धान": "Paddy(Dhan)", "मक्का": "Maize",
        "सरसों": "Mustard", "आलू": "Potato", "प्याज": "Onion",
        "टमाटर": "Tomato", "गन्ना": "Sugarcane", "चना": "Bengal Gram(Gram)(Whole)"
    }
    crop_en = crop_map.get(crop_hi, crop_hi)
    perishable_crops = ["Potato", "Onion", "Tomato", "Sugarcane", "आलू", "प्याज", "टमाटर", "गन्ना"]

    live_price = 0
    live_price_text = "आज का लाइव भाव उपलब्ध नहीं है।"
    
    try:
        url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        params = {"api-key": DATA_GOV_KEY, "format": "json", "limit": 1, "filters[state]": state, "filters[district]": district, "filters[commodity]": crop_en}
        
        async with httpx.AsyncClient(timeout=8) as client:
            res = await client.get(url, params=params)
            records = res.json().get("records", []) if res.status_code == 200 else []
            
            # फॉलबैक 1: पूरे राज्य में फसल खोजो
            if not records:
                params.pop("filters[district]")
                res2 = await client.get(url, params=params)
                records = res2.json().get("records", []) if res2.status_code == 200 else []

            if records:
                live_price = int(float(records[0].get("modal_price", "0")))
                actual_market = records[0].get("market", district)
                actual_crop = records[0].get("commodity", crop_hi)
                live_price_text = f"आज {actual_market} मंडी में {actual_crop} का लाइव भाव ₹{live_price} प्रति क्विंटल है।"
    except Exception as e:
        print(f"Agmarknet API Error: {e}")

    # ==========================================
    # 🧠 असली RULE-BASED सलाह (कोई AI नहीं)
    # ==========================================
    if user_price == 0 and live_price > 0:
        user_price = live_price
        
    decision = "HOLD"
    reason_1 = ""
    reason_2 = ""
    
    if live_price > 0 and user_price > 0:
        if user_price > live_price + 50:
            decision = "SELL"
            reason_1 = f"आपको ₹{user_price} मिल रहे हैं जो आज के मंडी भाव (₹{live_price}) से ज़्यादा है।"
            reason_2 = "इससे बेहतर मौका शायद ही मिले, इसलिए अभी बेच देना फायदेमंद है।"
        elif user_price < live_price - 50:
            decision = "HOLD"
            reason_1 = f"आपको ₹{user_price} मिल रहे हैं जो आज के मंडी भाव (₹{live_price}) से कम है।"
            if crop_en in perishable_crops:
                decision = "SELL"
                reason_2 = "लेकिन चूंकि यह जल्दी खराब होने वाली फसल है, इसलिए ज़्यादा इंतज़ार मत करें। जितना मिल रहा है उसमें बेच दें।"
            else:
                reason_2 = "आपको थोड़ा इंतज़ार करना चाहिए, भाव बढ़ने का इंतज़ार करें।"
        else:
            decision = "HOLD"
            reason_1 = f"आज का मंडी भाव ₹{live_price} के आस-पास है।"
            reason_2 = "भाव स्थिर है, थोड़ा और इंतज़ार किया जा सकता है।"
    else:
        if crop_en in perishable_crops:
            decision = "SELL"
            reason_1 = "यह जल्दी खराब होने वाली फसल है।"
            reason_2 = "लाइव भाव नहीं मिल पा रहा, पर इसे ज़्यादा दिन रोखना ठीक नहीं है।"
        else:
            decision = "HOLD"
            reason_1 = "लाइव भाव नहीं मिल पा रहा है।"
            reason_2 = "सूखी जगह में रखें और स्थानीय मंडी में भाव चेक करें।"

    # AI का काम सिर्फ इसे सुंदर हिंदी में बदलना है
    system = f"You are a translator. Reply ONLY in Hindi. Present this decision clearly: '{reason_1} {reason_2}' and end with 'यह AI सलाह है — मंडी में भाव की पुष्टि करें।'"
    
    try:
        final_reply = await groq_chat(system, "Translate this.", max_tokens=200)
    except Exception:
        final_reply = f"{reason_1} {reason_2} यह AI सलाह है — मंडी में भाव की पुष्टि करें।"

    return {
        "reply": final_reply,
        "live_price": live_price_text,
        "trend": "मंडी डेटा"
    }
