from __future__ import annotations

from datetime import datetime, timezone
from collections import Counter
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

import config

load_dotenv()


EDU_CONTENT: dict[str, dict[str, Any]] = {
    "Tornadoes": {
        "overview": "Tornadoes are violently rotating columns of air extending from thunderstorms to the ground.",
        "where": "Most common in the central and southeastern U.S. (including Tornado Alley and Dixie Alley), but possible in nearly every state.",
        "why": "They often form in supercells when warm humid air meets cooler drier air with strong vertical wind shear.",
        "warning_signs": [
            "Greenish or very dark sky",
            "Large hail with intense thunderstorm activity",
            "Loud, continuous roar similar to a freight train",
        ],
        "emergency": [
            "Go to a basement or a small interior room on the lowest floor.",
            "Stay away from windows and outside walls.",
            "Protect your head and neck with your arms, a helmet, or heavy blankets.",
            "If in a vehicle, seek sturdy shelter as quickly as possible.",
        ],
    },
    "Flash Floods": {
        "overview": "Flash floods can happen within minutes after intense rain and are among the deadliest weather hazards.",
        "where": "Common in cities, canyons, mountain valleys, burn-scar areas, and near small rivers or creeks.",
        "why": "Rainfall exceeds drainage capacity, and runoff rises rapidly where soil cannot absorb water.",
        "warning_signs": [
            "Rapidly rising water levels",
            "Water crossing roads or filling low-lying areas",
            "Heavy rain persisting upstream",
        ],
        "emergency": [
            "Move to higher ground immediately.",
            "Never drive through floodwater (Turn Around, Don't Drown).",
            "Avoid walking in moving water, even if it looks shallow.",
            "Follow local evacuation instructions without delay.",
        ],
    },
    "Severe Thunderstorms": {
        "overview": "Severe thunderstorms can produce damaging winds, hail, lightning, and sometimes tornadoes.",
        "where": "Frequent across much of the U.S., especially in spring and summer in the Plains, Midwest, and Southeast.",
        "why": "They form from unstable warm air, moisture, lift, and wind shear that intensify storm structure.",
        "warning_signs": [
            "Towering dark clouds",
            "Frequent lightning and loud thunder",
            "Sudden strong gusts or hail",
        ],
        "emergency": [
            "Go indoors as soon as thunder is heard.",
            "Avoid windows, corded electronics, and plumbing during lightning.",
            "Secure outdoor items if high wind is expected.",
            "Monitor warnings for rapid escalation to tornado risk.",
        ],
    },
    "Hurricanes": {
        "overview": "Hurricanes are tropical cyclones with sustained winds of 74 mph or more and major flood potential.",
        "where": "Most common along Atlantic and Gulf coasts, with rain and wind impacts often extending far inland.",
        "why": "They develop over warm ocean waters with organized low pressure, deep moisture, and supportive wind conditions.",
        "warning_signs": [
            "Official hurricane watches and warnings",
            "Increasing rain bands and sustained winds",
            "Storm surge and rapid coastal water rise",
        ],
        "emergency": [
            "Follow evacuation orders immediately.",
            "Prepare water, food, medications, lighting, and backup power.",
            "Expect outages and avoid floodwaters after landfall.",
            "If sheltering at home, stay in an interior room away from windows.",
        ],
    },
    "Wildfires": {
        "overview": "Wildfires are uncontrolled fires that spread through vegetation and can threaten lives, homes, and air quality.",
        "where": "Most common in dry and windy regions, especially in western North America, but risk is expanding.",
        "why": "Heat, dry fuels, low humidity, and strong winds make ignition and spread much more likely.",
        "warning_signs": [
            "Red Flag Warnings",
            "Visible smoke columns or ash",
            "Fast-changing wind conditions",
        ],
        "emergency": [
            "Evacuate early if advised.",
            "Keep go-bags ready and include masks for smoke.",
            "Use multiple alert channels for route updates.",
            "If sheltering temporarily, close windows and improve indoor air filtration.",
        ],
    },
    "Winter Storms": {
        "overview": "Winter storms combine snow, ice, strong winds, and dangerous cold.",
        "where": "Common in northern states, high elevations, and central/eastern regions during major cold outbreaks.",
        "why": "Cold air interacting with moisture systems can produce heavy snow, freezing rain, and blizzard conditions.",
        "warning_signs": [
            "Sharp temperature drops",
            "Onset of freezing rain or sleet",
            "Rapid visibility loss in blowing snow",
        ],
        "emergency": [
            "Avoid non-essential travel during warnings.",
            "Keep emergency vehicle supplies ready.",
            "Prepare for power loss and safe indoor heating.",
            "Watch for ice-loaded trees and falling limbs.",
        ],
    },
}

FALLBACK_KB: dict[str, str] = {
    "watch warning": (
        "A watch means conditions are favorable. A warning means severe weather is happening or imminent. "
        "Warnings require immediate protective action."
    ),
    "tornado safety": (
        "Go to the lowest interior room, avoid windows, protect your head, and stay informed with official alerts."
    ),
    "flood safety": "Avoid flood waters and never drive through flooded roads. Move to higher ground quickly.",
    "hurricane prep": "Prepare water, food, medications, power backups, and evacuation plans before landfall.",
}


def app_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at top left, #f0f9ff 0%, #f8fafc 45%, #eef2ff 100%);
            }
            .main-title {
                font-size: 3.2rem;
                font-weight: 800;
                margin-bottom: 0.3rem;
                background: linear-gradient(90deg, #0f766e, #2563eb, #7c3aed);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .subtitle { color: #475569; margin-bottom: 1.2rem; font-size: 1.1rem; }
            .card {
                border: 1px solid #dbeafe; border-radius: 16px; padding: 1rem 1.1rem;
                background: linear-gradient(180deg, #ffffff, #f8fbff); margin-bottom: 0.9rem;
                box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
            }
            .insight {
                border: 1px solid #bfdbfe; border-radius: 12px; padding: 0.85rem 1rem;
                background: linear-gradient(180deg, #eff6ff, #f8fafc); margin-bottom: 0.75rem;
            }
            .section-title {
                font-size: 1.9rem;
                font-weight: 700;
                margin: 0.4rem 0 0.8rem 0;
                color: #0f172a;
            }
            .hero-badge {
                display: inline-block;
                border-radius: 999px;
                background: rgba(37, 99, 235, 0.10);
                border: 1px solid rgba(37, 99, 235, 0.28);
                color: #1d4ed8;
                padding: 0.22rem 0.65rem;
                font-size: 0.86rem;
                font-weight: 600;
                margin-bottom: 0.7rem;
            }
            div[role="radiogroup"] {
                gap: 0.6rem;
                padding: 0.2rem 0 0.8rem 0;
            }
            div[role="radiogroup"] > label {
                border: 1px solid #cbd5e1;
                border-radius: 12px;
                padding: 0.75rem 1rem;
                background: #ffffff;
                font-weight: 600;
                font-size: 1.02rem;
                min-width: 180px;
                justify-content: center;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
            }
            div[role="radiogroup"] > label:has(input:checked) {
                background: linear-gradient(90deg, #0f766e, #2563eb);
                color: white;
                border-color: #0f766e;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_openai_client() -> OpenAI | None:
    if not config.OPENAI_API_KEY:
        return None
    return OpenAI(api_key=config.OPENAI_API_KEY)


@st.cache_data(ttl=300)
def fetch_active_alerts() -> list[dict[str, Any]]:
    response = requests.get(config.NOAA_ALERTS_URL, headers=config.HEADERS, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return []
    features = payload.get("features", [])
    if not isinstance(features, list):
        return []
    return features


def to_alert_rows(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        if not isinstance(props, dict):
            props = {}
        if not isinstance(geom, dict):
            geom = {}
        coordinates = geom.get("coordinates")
        lon, lat = (None, None)
        if geom.get("type") == "Point" and isinstance(coordinates, list) and len(coordinates) == 2:
            lon, lat = coordinates[0], coordinates[1]

        geocode = props.get("geocode") or {}
        if not isinstance(geocode, dict):
            geocode = {}
        ugc = geocode.get("UGC", ["Unknown"])
        if not isinstance(ugc, list) or not ugc:
            ugc = ["Unknown"]

        rows.append(
            {
                "id": props.get("id", ""),
                "event": props.get("event", "Unknown"),
                "severity": props.get("severity", "Unknown"),
                "urgency": props.get("urgency", "Unknown"),
                "area": props.get("areaDesc", "Unknown"),
                "state": str(ugc[0])[:2],
                "headline": props.get("headline", ""),
                "instruction": props.get("instruction", ""),
                "effective": props.get("effective", ""),
                "expires": props.get("expires", ""),
                "lat": lat,
                "lon": lon,
            }
        )
    return rows


def extract_state(area: str) -> str:
    chunks = [part.strip() for part in area.split(";")]
    if not chunks:
        return "Unknown"
    right = chunks[0].split(",")
    if len(right) > 1:
        return right[-1].strip()[:2].upper()
    return "Unknown"


def route_risk_by_state(rows: list[dict[str, Any]], route_states: list[str]) -> list[dict[str, Any]]:
    state_counts = Counter()
    for row in rows:
        state_counts[extract_state(str(row.get("area", "Unknown")))] += 1

    result: list[dict[str, Any]] = []
    for state in route_states:
        count = int(state_counts.get(state.upper(), 0))
        risk = "High" if count >= 10 else ("Medium" if count >= 4 else "Low")
        result.append({"state": state.upper(), "active_alerts": count, "risk": risk})
    return result


def fallback_chat_answer(question: str) -> str:
    q = question.lower()
    for key, answer in FALLBACK_KB.items():
        if key in q:
            return answer
    return (
        "I specialize in historical severe-weather facts, safety evidence, and definitions "
        "(for example: watch vs warning, historic tornado patterns, and preparedness basics). "
        "Try asking: 'Historically, what conditions are common in major U.S. tornado outbreaks?'"
    )


def llm_chat_answer(client: OpenAI | None, question: str, context: str) -> str:
    if client is None:
        return fallback_chat_answer(question)
    try:
        completion = client.chat.completions.create(
            model=config.CHATBOT_MODEL,
            temperature=0.2,
            max_tokens=350,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a weather education assistant. Be clear, practical, and safety-first. "
                        "You specialize in historical weather facts and evidence-based explanations. "
                        "Do not fabricate real-time warnings; defer to NOAA/NWS for urgent actions."
                    ),
                },
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
            ],
        )
        return completion.choices[0].message.content or fallback_chat_answer(question)
    except Exception:
        return fallback_chat_answer(question)


def render_dashboard(rows: list[dict[str, Any]]) -> None:
    st.markdown("<div class='section-title'>Real-Time Weather Alerts</div>", unsafe_allow_html=True)
    if not rows:
        st.warning("No active severe alerts available at the moment.")
        return

    events = [str(row.get("event", "Unknown")) for row in rows]
    severities = [str(row.get("severity", "Unknown")) for row in rows]
    areas = [str(row.get("area", "Unknown")) for row in rows]

    c1, c2, c3 = st.columns(3)
    c1.metric("Active Alerts", len(rows))
    c2.metric("Unique Events", len(set(events)))
    c3.metric("Areas Impacted", len(set(areas)))

    st.markdown('<div class="card">Filter and inspect current NOAA severe alerts.</div>', unsafe_allow_html=True)
    event_filter = st.multiselect("Event types", sorted(set(events)))
    severity_filter = st.multiselect("Severities", sorted(set(severities)))

    filtered = rows[:]
    if event_filter:
        filtered = [r for r in filtered if r.get("event") in event_filter]
    if severity_filter:
        filtered = [r for r in filtered if r.get("severity") in severity_filter]

    st.markdown("#### Notable Alerts")
    for row in filtered[:10]:
        st.markdown(
            f"- **{row.get('event', 'Unknown')}** | {row.get('severity', 'Unknown')} | "
            f"{row.get('urgency', 'Unknown')} | {row.get('area', 'Unknown')}"
        )
    if len(filtered) > 10:
        st.caption(f"Showing 10 of {len(filtered)} alerts to keep the display concise.")

    event_counts = Counter(str(r.get("event", "Unknown")) for r in filtered)
    st.caption("Event mix story: which hazards are driving today's alert landscape.")
    for event, count in event_counts.most_common(10):
        st.progress(min(count / max(len(filtered), 1), 1.0), text=f"{event}: {count}")

    map_points = [r for r in filtered if r.get("lat") is not None and r.get("lon") is not None]
    if map_points:
        st.caption("Geographic spread of point-based alerts.")
        st.map(
            [{"lat": p["lat"], "lon": p["lon"]} for p in map_points],
            latitude="lat",
            longitude="lon",
        )


def render_routes(rows: list[dict[str, Any]]) -> None:
    st.markdown("<div class='section-title'>Safer Route Recommendation</div>", unsafe_allow_html=True)
    st.write("Enter a route as states (e.g., `TX, OK, KS`) and compare weather risk by active alerts.")
    raw_route = st.text_input("Route states (comma-separated)", value="TX, OK, KS")
    route_states = [s.strip().upper() for s in raw_route.split(",") if s.strip()]

    if not route_states:
        st.info("Please enter at least one state code.")
        return

    risk_rows = route_risk_by_state(rows, route_states)
    for r in risk_rows:
        st.markdown(f"- **{r['state']}**: {r['active_alerts']} active alerts ({r['risk']} risk)")
        st.progress(min(r["active_alerts"] / 15.0, 1.0), text=f"{r['state']} risk signal")

    low_risk = [r["state"] for r in sorted(risk_rows, key=lambda x: x["active_alerts"])]
    st.success(f"Recommended order (lower active-alert exposure first): {' -> '.join(low_risk)}")


def render_education() -> None:
    st.markdown("<div class='section-title'>Learn: Severe Weather and Precautions</div>", unsafe_allow_html=True)
    topic = st.selectbox("Choose a topic", list(EDU_CONTENT.keys()))
    content = EDU_CONTENT[topic]
    st.markdown(f"### {topic}")
    st.write(content["overview"])
    st.markdown(f"**Where it most often happens:** {content['where']}")
    st.markdown(f"**Why it happens:** {content['why']}")
    st.markdown("#### Common warning signs")
    for sign in content["warning_signs"]:
        st.markdown(f"- {sign}")
    st.markdown("#### What to do in an emergency")
    for step in content["emergency"]:
        st.markdown(f"- {step}")


def render_chatbot(rows: list[dict[str, Any]]) -> None:
    st.markdown("<div class='section-title'>Historical Weather Facts Chatbot</div>", unsafe_allow_html=True)
    st.caption(
        "This chatbot is specialized for historical weather facts, patterns, and safety context "
        "(not live emergency dispatch)."
    )
    client = get_openai_client()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_input("Ask a historical weather question")
    if st.button("Send", type="primary") and question.strip():
        context = (
            f"Current active severe alerts: {len(rows)}. "
            f"Top event types: {', '.join([k for k, _ in Counter([str(r.get('event', 'Unknown')) for r in rows]).most_common(3)]) if rows else 'N/A'}."
        )
        answer = llm_chat_answer(client, question.strip(), context)
        st.session_state.chat_history.append({"q": question.strip(), "a": answer})

    for item in reversed(st.session_state.chat_history[-8:]):
        st.markdown(f"<div class='card'><b>You:</b> {item['q']}<br/><b>Bot:</b> {item['a']}</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title=config.PAGE_TITLE, layout=config.LAYOUT)
    app_css()
    st.markdown("<div class='hero-badge'>Live NOAA + Practical AI Insights</div>", unsafe_allow_html=True)
    st.markdown('<div class="main-title">TwistEd Weather Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Learn severe weather, monitor active alerts, and explore lower-risk travel paths.</div>',
        unsafe_allow_html=True,
    )
    st.warning(config.SAFETY_DISCLAIMER)
    st.markdown("### Explore Sections")
    section = st.radio(
        "Navigation",
        [
            "Live Alerts",
            "Safer Routes",
            "Historical Chatbot",
            "Learn",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    try:
        rows = to_alert_rows(fetch_active_alerts())
    except Exception as exc:
        st.error(f"Could not fetch NOAA alerts right now: {exc}")
        rows = []

    if rows:
        now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        st.caption(f"Last refreshed: {now}")

    if section == "Live Alerts":
        render_dashboard(rows)
    elif section == "Safer Routes":
        render_routes(rows)
    elif section == "Historical Chatbot":
        render_chatbot(rows)
    elif section == "Learn":
        render_education()


if __name__ == "__main__":
    main()
