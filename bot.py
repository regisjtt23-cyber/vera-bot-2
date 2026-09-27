"""Stateful, deterministic Vera challenge API."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("vera")
STARTED = time.monotonic()
MAX_CONTEXT_BYTES = 500 * 1024
MAX_ACTIONS = 20

app = FastAPI(title="Vera Merchant Growth Assistant", version="1.0.0")
lock = RLock()
contexts: dict[tuple[str, str], dict[str, Any]] = {}
conversations: dict[str, dict[str, Any]] = {}
sent_suppression: set[str] = set()
sent_content: set[str] = set()
auto_reply_counts: dict[tuple[str, str], int] = {}
blocked_contacts: set[tuple[str, str]] = set()

CUSTOMER_TRIGGER_KINDS = {
    "recall_due", "appointment_tomorrow", "trial_followup",
    "wedding_package_followup", "customer_lapsed_soft", "customer_lapsed_hard",
    "chronic_refill_due", "winback_eligible",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _urgency(trigger: dict) -> int:
    try:
        return max(1, min(5, int(trigger.get("urgency", 1))))
    except (TypeError, ValueError, OverflowError):
        return 1


def _contains_phrase(message: str, phrase: str) -> bool:
    words = r"\s+".join(re.escape(part) for part in phrase.lower().split())
    return re.search(rf"(?<!\w){words}(?!\w)", message) is not None


class ContextPush(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str = Field(min_length=1, max_length=256)
    version: int = Field(ge=1)
    payload: dict[str, Any]
    delivered_at: str


class TickRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    now: str
    available_triggers: list[str] = Field(default_factory=list, max_length=1000)

    @field_validator("now")
    @classmethod
    def valid_now(cls, value: str) -> str:
        if _parse_time(value) is None:
            raise ValueError("now must be an ISO-8601 datetime")
        return value


class ReplyRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    conversation_id: str = Field(min_length=1, max_length=256)
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str = "merchant"
    message: str = Field(min_length=1, max_length=10000)
    received_at: str | None = None
    turn_number: int = Field(default=1, ge=1)

    @field_validator("from_role")
    @classmethod
    def valid_role(cls, value: str) -> str:
        if value not in {"merchant", "customer"}:
            raise ValueError("from_role must be merchant or customer")
        return value


def _context(scope: str, key: str) -> dict[str, Any] | None:
    item = contexts.get((scope, key))
    return item["payload"] if item else None


def _name(merchant: dict) -> str:
    return (merchant.get("identity") or {}).get("owner_first_name") or (merchant.get("identity") or {}).get("name") or "there"


def _business(merchant: dict) -> str:
    return (merchant.get("identity") or {}).get("name") or "your business"


def _category(merchant: dict) -> str:
    return str(merchant.get("category_slug") or "").lower()


def _language_preference(profile: dict | None) -> str:
    profile = profile or {}
    identity = profile.get("identity") or {}
    preferences = profile.get("preferences") or {}
    value = identity.get("language_pref") or identity.get("preferred_language") or preferences.get("language_pref") or ""
    if not value:
        languages = identity.get("languages") or []
        value = ",".join(str(x) for x in languages)
    return str(value).lower()


def _localized_cta(body: str, profile: dict | None, audience: str = "customer") -> str:
    """Add a restrained Hindi-English CTA only when the supplied preference asks for it."""
    language = _language_preference(profile)
    if not any(token in language for token in ("hindi", "hinglish", "hi-en", "hi_en", "हिंदी")):
        return body
    swaps = {
        "Would you like us to help find a suitable appointment time?": "Suitable appointment time chahiye?",
        "Would you like us to share the next-session options?": "Next-session options share karein?",
        "Would you like to discuss the next bridal appointment?": "Next bridal appointment discuss karein?",
        "Would you like to hear about a suitable option?": "Suitable option ke baare mein batayein?",
        "Please reply if you need to change the time.": "Time change karna ho toh reply karein.",
        "Reply if you'd like us to check availability for you.": "Availability check karni ho toh reply karein.",
    }
    if audience == "merchant":
        swaps.update({
            "Want me to draft a match-day delivery message?": "Match-day delivery message draft karoon?",
            "Want me to organize the next steps?": "Next steps organize karoon?",
            "Want me to organize the provided details?": "Provided details organize karoon?",
            "Want to review how your listing can stand out?": "Listing ko kaise alag dikhayein, review karein?",
            "Want me to isolate the first metric to check?": "Pehle check karne wala metric identify karoon?",
            "Want one practical adjustment to try first?": "Ek practical adjustment try karein?",
            "Want me to pinpoint the service window to review first?": "Review karne ke liye service window identify karoon?",
            "Want one coaching-style retention step to review?": "Retention ke liye ek coaching-style step review karein?",
            "Want me to summarize the change for your team?": "Team ke liye change summarize karoon?",
        })
    for source, replacement in swaps.items():
        if source in body:
            return body.replace(source, replacement)
    return body


def _slot_labels(slots: list, customer: dict, limit: int = 2) -> list[str]:
    preference = str((customer.get("preferences") or {}).get("preferred_slots") or "").lower()
    ranked = []
    for slot in slots:
        if not isinstance(slot, dict) or not slot.get("label"):
            continue
        label = str(slot["label"])
        normalized = label.lower()
        iso = str(slot.get("iso") or "")
        hour = None
        try:
            parsed = _parse_time(iso)
            hour = parsed.hour if parsed else None
        except (TypeError, ValueError):
            pass
        matches = True
        if "evening" in preference:
            matches = ("pm" in normalized) or (hour is not None and hour >= 17)
        elif "morning" in preference:
            matches = ("am" in normalized) or (hour is not None and hour < 12)
        elif "afternoon" in preference:
            matches = hour is not None and 12 <= hour < 17
        if "weekday" in preference:
            matches = matches and not any(day in normalized for day in ("sat", "sun"))
        if "weekend" in preference:
            matches = matches and any(day in normalized for day in ("sat", "sun"))
        ranked.append((not matches, label))
    ranked.sort(key=lambda item: item[0])
    return [label for _, label in ranked[:limit]]


def _active_offer(merchant: dict, terms: tuple[str, ...] = ()) -> str | None:
    for offer in merchant.get("offers") or []:
        if str(offer.get("status", "active")).lower() != "active":
            continue
        title = str(offer.get("title") or "").strip()
        if title and (not terms or any(term in title.lower() for term in terms)):
            return title
    return None


def _fmt_pct(value: Any) -> str | None:
    try:
        return f"{abs(float(value)) * 100:g}%"
    except (ValueError, TypeError):
        return None


def _digest_item(category: dict, trigger: dict) -> dict | None:
    payload = trigger.get("payload") or {}
    ident = next((payload.get(k) for k in ("top_item_id", "digest_item_id", "alert_id", "item_id") if payload.get(k)), None)
    for item in category.get("digest") or []:
        if ident and item.get("id") == ident:
            return item
    return None


def _signal(trigger: dict, category: dict, merchant: dict, customer: dict | None) -> dict[str, Any]:
    """Return one grounded, ranked signal with its compact generation facts."""
    kind = str(trigger.get("kind") or "unknown")
    p = trigger.get("payload") or {}
    cat = _category(merchant)
    expected_categories = {
        "wedding_package_followup": "salons", "ipl_match_today": "restaurants",
        "chronic_refill_due": "pharmacies", "supply_alert": "pharmacies",
    }
    if kind in expected_categories and cat != expected_categories[kind]:
        return {"score": -1, "body": "", "cta": "none", "facts": [], "label": kind}
    who = _name(merchant)
    biz = _business(merchant)
    ident = merchant.get("identity") or {}
    locality = ident.get("locality")
    c_name = ((customer or {}).get("identity") or {}).get("name")
    urgency = _urgency(trigger)
    facts: list[str] = []
    score = urgency * 5
    if kind == "research_digest":
        item = _digest_item(category, trigger)
        if item:
            title, source = item.get("title"), item.get("source")
            if not title:
                return {"score": -1, "body": "", "cta": "none", "facts": [], "label": kind}
            if title: facts.append(str(title).rstrip("."))
            if item.get("trial_n"): facts.append(f"{item['trial_n']:,}-participant trial")
            if source: facts.append(str(source))
            summary = str(item.get("summary") or "")
            cohort = (merchant.get("customer_aggregate") or {}).get("high_risk_adult_count")
            body = f"{who}, {str(item.get('kind') or 'category').replace('_', ' ')} update for {biz}: {title}."
            if item.get("trial_n") and summary:
                body += f" A {item['trial_n']:,}-participant trial reports: {summary}"
                facts.append(summary)
            elif summary:
                body += f" {summary}"
                facts.append(summary)
            if cohort:
                body += f" This may be relevant to your {cohort} high-risk adult patients."
                facts.append(str(cohort))
            if source: body += f" Source: {source}."
            body += " Want me to pull the key points for you?"
            score += 30 + (8 if item.get("trial_n") else 0)
            return {"score": score, "body": body, "cta": "open_ended", "facts": facts, "label": kind}
    if kind in {"recall_due", "appointment_tomorrow", "trial_followup", "wedding_package_followup", "customer_lapsed_soft", "customer_lapsed_hard", "chronic_refill_due", "winback_eligible"} and customer:
        consent = customer.get("consent") or {}
        prefs = customer.get("preferences") or {}
        scopes = [str(x).lower() for x in (consent.get("scope") or [])]
        opted = bool(consent.get("opted_in_at"))
        if kind in {"customer_lapsed_soft", "customer_lapsed_hard", "winback_eligible"}:
            opted = opted and prefs.get("promotion_opt_in", prefs.get("promotional_opt_in", True)) is not False
        else:
            opted = opted and prefs.get("reminder_opt_in", True) is not False
        channel = str(prefs.get("channel") or "whatsapp").lower()
        if "whatsapp" not in channel and channel not in {"any", "all"}:
            opted = False
        required_scopes = {
            "recall_due": {"recall_reminders", "appointment_reminders"},
            "appointment_tomorrow": {"appointment_reminders"},
            "chronic_refill_due": {"refill_reminders"},
            "trial_followup": {"appointment_reminders", "program_updates", "kids_program_updates"},
            "wedding_package_followup": {"bridal_package_followup", "appointment_reminders"},
            "customer_lapsed_soft": {"promotional_offers", "winback_offers"},
            "customer_lapsed_hard": {"promotional_offers", "winback_offers"},
            "winback_eligible": {"promotional_offers", "winback_offers"},
        }.get(kind, set())
        if not opted or (required_scopes and not required_scopes.intersection(scopes)):
            return {"score": -1, "body": "", "cta": "none", "facts": [], "label": kind}
        if cat == "pharmacies" and kind == "chronic_refill_due":
            meds = p.get("molecule_list") or []
            date = p.get("stock_runs_out_iso")
            label = ", ".join(str(x) for x in meds[:3])
            body = f"Hi {c_name or 'there'}, {biz} here. Your refill reminder is due"
            if label: body += f" for {label}"
            if date: body += f" by {str(date)[:10]}"
            body += ". Reply if you'd like us to check availability for you."
            facts.extend([*meds[:3], str(date or "")])
            return {"score": score + 36, "body": body, "cta": "open_ended", "facts": facts, "label": kind, "send_as": "merchant_on_behalf"}
        if kind == "recall_due":
            due = p.get("due_date")
            service = str(p.get("service_due") or p.get("service") or "follow-up")
            offer = _active_offer(merchant, ("clean", "recall")) if cat == "dentists" else None
            body = f"Hi {c_name or 'there'}, {biz} here. Your {service.replace('_', ' ')} reminder is coming up"
            if due: body += f" ({str(due)[:10]})"
            body += "."
            if offer: body += f" We have {offer}."
            slots = p.get("available_slots") or []
            labels = _slot_labels(slots, customer)
            if labels: body += " Available times: " + " or ".join(labels) + "."
            body += " Would you like us to help find a suitable appointment time?"
            body = _localized_cta(body, customer)
            facts.extend([str(due or ""), offer or "", *labels])
            return {"score": score + 38, "body": body, "cta": "open_ended", "facts": facts, "label": kind, "send_as": "merchant_on_behalf"}
        if kind == "appointment_tomorrow":
            body = f"Hi {c_name or 'there'}, a reminder from {biz}: your appointment is tomorrow. Please reply if you need to change the time."
            body = _localized_cta(body, customer)
            return {"score": score + 40, "body": body, "cta": "open_ended", "facts": [], "label": kind, "send_as": "merchant_on_behalf"}
        if kind == "trial_followup":
            body = f"Hi {c_name or 'there'}, {biz} here. Thanks for joining the trial"
            if p.get("trial_date"): body += f" on {str(p['trial_date'])[:10]}"
            body += ". Would you like us to share the next-session options?"
            body = _localized_cta(body, customer)
            return {"score": score + 28, "body": body, "cta": "open_ended", "facts": [str(p.get("trial_date") or "")], "label": kind, "send_as": "merchant_on_behalf"}
        if kind == "wedding_package_followup":
            date = p.get("wedding_date") or (customer.get("preferences") or {}).get("wedding_date")
            body = f"Hi {c_name or 'there'}, {biz} here. Following up on your bridal trial"
            if p.get("trial_completed"): body += f" from {str(p['trial_completed'])[:10]}"
            if date: body += f" — your wedding date is {str(date)[:10]}"
            body += ". Would you like to discuss the next bridal appointment?"
            body = _localized_cta(body, customer)
            return {"score": score + 34, "body": body, "cta": "open_ended", "facts": [str(date or ""), str(p.get("trial_completed") or "")], "label": kind, "send_as": "merchant_on_behalf"}
        if kind in {"customer_lapsed_soft", "customer_lapsed_hard", "winback_eligible"}:
            days = p.get("days_since_last_visit")
            body = f"Hi {c_name or 'there'}, {biz} here. We'd be glad to welcome you back whenever it suits you"
            if days is not None: body += f"; it has been {days} days since your last visit"
            focus = p.get("previous_focus")
            if focus: body += f". If {str(focus).replace('_', ' ')} is still your focus, we can share current options"
            body += ". Would you like to hear about a suitable option?"
            body = _localized_cta(body, customer)
            return {"score": score + 24, "body": body, "cta": "open_ended", "facts": [str(days or "")], "label": kind, "send_as": "merchant_on_behalf"}
    # Merchant-facing deterministic trigger policies.
    if kind == "active_planning_intent":
        topic = str(p.get("intent_topic") or "the plan")
        body = f"{who}, picking up your {topic.replace('_', ' ')} idea: I can help shape a first draft using the details you provide. What quantity or format should I start with?"
        return {"score": score + 45, "body": body, "cta": "open_ended", "facts": [topic], "label": kind}
    if kind == "curious_ask_due":
        body = f"Hi {who}, quick question for {biz}: which service has customers asked you about most this week? I can turn your answer into a short post draft."
        return {"score": score + 25, "body": body, "cta": "open_ended", "facts": [], "label": kind}
    if kind in {"perf_dip", "seasonal_perf_dip", "perf_spike"}:
        metric = str(p.get("metric") or "views")
        pct = _fmt_pct(p.get("delta_pct"))
        delta = p.get("delta_pct")
        direction = "down" if isinstance(delta, (int, float)) and delta < 0 else "up" if isinstance(delta, (int, float)) and delta > 0 else ""
        body = f"{who}, your {metric}"
        if direction:
            body += f" are {direction} {pct}"
        else:
            body += " changed in the latest reported window"
        if p.get("window"): body += f" over {p['window']}"
        latest = (merchant.get("performance") or {}).get(metric)
        if isinstance(latest, (int, float)) and not isinstance(latest, bool):
            body += f"; latest reported value {latest:g}"
            peer = category.get("peer_stats") or {}
            if metric.lower() in {"ctr", "click_through_rate"} and isinstance(peer.get("avg_ctr"), (int, float)):
                body += f" vs category average {peer['avg_ctr']:g}"
        if kind == "seasonal_perf_dip" and p.get("season_note"):
            body += f". The context flags this as seasonal ({str(p['season_note']).replace('_', ' ')})"
        if direction == "down":
            category_cta = {
                "dentists": "Want me to isolate the first metric to check?",
                "salons": "Want one practical adjustment to try first?",
                "restaurants": "Want me to pinpoint the service window to review first?",
                "gyms": "Want one coaching-style retention step to review?",
                "pharmacies": "Want me to summarize the change for your team?",
            }
        else:
            category_cta = {
                "dentists": "Want me to identify the reported change's likely driver?",
                "salons": "Want a quick read on the reported lift's likely driver?",
                "restaurants": "Want me to find a service window where this lift can be repeated?",
                "gyms": "Want one way to build on this reported lift?",
                "pharmacies": "Want me to summarize this reported change for your team?",
            }
        body += ". " + category_cta.get(cat, "Want me to identify one next step from this change?")
        return {"score": score + 34 + (5 if p.get("delta_pct") is not None else 0), "body": body, "cta": "open_ended", "facts": [str(p.get("delta_pct") or ""), metric, str(p.get("window") or "")], "label": kind}
    if kind == "ipl_match_today":
        match, tm = p.get("match"), p.get("match_time_iso")
        offer = _active_offer(merchant, ("pizza", "bogo", "offer"))
        body = f"{who}, {match or 'the match'} is on"
        if tm: body += f" at {str(tm)[11:16]}"
        body += " today"
        if offer: body += f"; your active offer is {offer}"
        body += ". Want me to draft a match-day delivery message?"
        return {"score": score + 32, "body": body, "cta": "open_ended", "facts": [str(match or ""), str(tm or ""), offer or ""], "label": kind}
    if kind in {"supply_alert", "regulation_change", "cde_opportunity"}:
        item = _digest_item(category, trigger)
        if item:
            title, source = str(item.get("title") or ""), item.get("source")
            if kind == "supply_alert":
                batches = p.get("affected_batches") or []
                body = f"{who}, supply alert: {title}"
                if p.get("molecule"): body += f" ({p['molecule']})"
                if batches: body += f"; affected batches listed: {', '.join(str(x) for x in batches)}"
                if p.get("manufacturer"): body += f"; manufacturer: {p['manufacturer']}"
                facts.extend([str(x) for x in batches])
            else:
                body = f"{who}, a {str(item.get('kind') or 'category').replace('_', ' ')} update: {title}"
            if item.get("summary"): body += f". {item['summary']}"
            if item.get("actionable"): body += f" {item['actionable']}"
            if source: body += f" ({source})"
            body += ". Want me to organize the next steps?"
            return {"score": score + 42, "body": body, "cta": "open_ended", "facts": [title, str(source or "")], "label": kind}
        details = []
        for k in ("molecule", "manufacturer", "deadline_iso", "fee", "credits"):
            if p.get(k) is not None: details.append(f"{k.replace('_', ' ')}: {p[k]}")
        for k in ("affected_batches",):
            if p.get(k): details.extend(str(v) for v in p[k])
        body = f"{who}, {kind.replace('_', ' ')} information was flagged"
        if details: body += ": " + ", ".join(details)
        body += ". Please verify the source notice before acting. Want me to organize the provided details?"
        return {"score": score + 38, "body": body, "cta": "open_ended", "facts": details, "label": kind}
    if kind == "competitor_opened":
        detail = p.get("competitor_name")
        distance = p.get("distance_km")
        body = f"{who}, a competitor update"
        if detail: body += f": {detail}"
        if distance is not None: body += f" is {distance} km away"
        if p.get("their_offer"): body += f" and lists {p['their_offer']}"
        body += ". Want to review how your listing can stand out?"
        return {"score": score + 35, "body": body, "cta": "open_ended", "facts": [str(detail or ""), str(distance or ""), str(p.get("their_offer") or "")], "label": kind}
    if kind == "review_theme_emerged":
        body = f"{who}, recent reviews flagged {p.get('occurrences_30d', 'a')} mentions of {str(p.get('theme') or 'a recurring theme').replace('_', ' ')}"
        if p.get("trend"): body += f" ({p['trend']})"
        body += ". Want me to help draft a response or a service-improvement checklist?"
        return {"score": score + 32, "body": body, "cta": "open_ended", "facts": [str(p.get("occurrences_30d") or ""), str(p.get("theme") or "")], "label": kind}
    if kind == "renewal_due":
        body = f"{who}, your {p.get('plan') or 'subscription'} renewal is approaching"
        if p.get("days_remaining") is not None: body += f" in {p['days_remaining']} days"
        if p.get("renewal_amount") is not None: body += f" (₹{p['renewal_amount']})"
        body += ". Want me to help review the renewal details?"
        return {"score": score + 34, "body": body, "cta": "open_ended", "facts": [str(p.get("days_remaining") or ""), str(p.get("renewal_amount") or "")], "label": kind}
    if kind == "milestone_reached":
        body = f"{who}, {p.get('metric', 'your metric')} is at {p.get('value_now', 'a new milestone')}"
        if p.get("milestone_value"): body += f"; next milestone is {p['milestone_value']}"
        body += ". Want me to draft a short customer thank-you post?"
        return {"score": score + 24, "body": body, "cta": "open_ended", "facts": [str(p.get("value_now") or ""), str(p.get("milestone_value") or "")], "label": kind}
    if kind == "festival_upcoming":
        days = p.get("days_until")
        if isinstance(days, (int, float)) and days > 30:
            return {"score": -1, "body": "", "cta": "none", "facts": [], "label": kind}
        festival = str(p.get("festival") or "upcoming festival")
        body = f"{who}, {festival} is"
        if p.get("date"): body += f" listed for {str(p['date'])[:10]}"
        if days is not None: body += f" ({days} days away)"
        body += f"; it may be relevant to {biz}"
        if locality: body += f" in {locality}"
        body += ". Want one practical way to prepare?"
        return {"score": score + 20, "body": body, "cta": "open_ended", "facts": [festival, str(p.get("date") or ""), str(days or "")], "label": kind}
    if kind == "winback_eligible":
        days = p.get("days_since_expiry")
        count = p.get("lapsed_customers_added_since_expiry")
        dip = _fmt_pct(p.get("perf_dip_pct"))
        body = f"{who}, your winback context"
        if days is not None: body += f" is {days} days past plan expiry"
        if count is not None: body += f" and lists {count} additional lapsed customers"
        if dip and isinstance(p.get("perf_dip_pct"), (int, float)) and p["perf_dip_pct"] < 0: body += f"; the reported {p.get('metric') or 'performance metric'} is down {dip}"
        body += ". Want a short winback message for this group?"
        return {"score": score + 26, "body": body, "cta": "open_ended", "facts": [str(days or ""), str(count or ""), str(dip or "")], "label": kind}
    if kind == "dormant_with_vera":
        days = p.get("days_since_last_merchant_message")
        topic = str(p.get("last_topic") or "the last business topic").replace("_", " ")
        body = f"{who}, it has been {days} days since our last merchant message" if days is not None else f"{who}, we haven't picked up our last conversation"
        body += f" about {topic}. Want to continue from there?"
        return {"score": score + 14, "body": body, "cta": "open_ended", "facts": [str(days or ""), topic], "label": kind}
    if kind == "gbp_unverified":
        if (merchant.get("identity") or {}).get("verified") is True or p.get("verified") is True:
            return {"score": -1, "body": "", "cta": "none", "facts": [], "label": kind}
        uplift = _fmt_pct(p.get("estimated_uplift_pct"))
        path = str(p.get("verification_path") or "")
        body = f"{who}, your listing is still marked unverified"
        if uplift: body += f"; the trigger estimates {uplift} potential uplift from verification"
        if path: body += f". The listed verification path is {path.replace('_', ' ')}"
        body += ". Want the setup steps?"
        return {"score": score + 28, "body": body, "cta": "open_ended", "facts": [str(uplift or ""), path], "label": kind}
    if kind == "category_seasonal":
        season = str(p.get("season") or "current season").replace("_", " ")
        trends = p.get("trends") or []
        trend = str(trends[0]).replace("_", " ") if trends else ""
        body = f"{who}, the category context flags {season}"
        if trend: body += f"; one reported signal is {trend}"
        if p.get("shelf_action_recommended") is True:
            body += ". A shelf-mix review is suggested"
        body += ". Want me to turn the supplied trend into one practical check?"
        return {"score": score + 22, "body": body, "cta": "open_ended", "facts": [season, trend], "label": kind}
    if kind == "scheduled_recurring":
        ask = str(p.get("ask_template") or "")
        if not ask:
            return {"score": -1, "body": "", "cta": "none", "facts": [], "label": kind}
        body = f"{who}, {ask.strip().rstrip('?')}? I can shape your answer into a short, ready-to-review draft."
        return {"score": score + 12, "body": body, "cta": "open_ended", "facts": [ask], "label": kind}
    if kind == "weather_heatwave":
        temperature = next((p.get(k) for k in ("temperature_c", "temp_c", "forecast_max_c") if p.get(k) is not None), None)
        day = p.get("date") or p.get("forecast_date")
        if temperature is None and not day:
            return {"score": -1, "body": "", "cta": "none", "facts": [], "label": kind}
        body = f"{who}, the pushed heat alert"
        if temperature is not None: body += f" lists {temperature}°C"
        if day: body += f" for {str(day)[:10]}"
        if locality: body += f" in {locality}"
        body += f". Want one {('customer-safety' if cat in {'dentists', 'pharmacies'} else 'operational')} check based on this alert?"
        return {"score": score + 22, "body": body, "cta": "open_ended", "facts": [str(temperature or ""), str(day or "")], "label": kind}
    if kind == "local_news_event":
        event = str(p.get("event") or p.get("title") or p.get("topic") or "")
        if not event:
            return {"score": -1, "body": "", "cta": "none", "facts": [], "label": kind}
        body = f"{who}, the local update is {event}"
        if p.get("impact_window"): body += f" ({p['impact_window']})"
        if p.get("date"): body += f" on {str(p['date'])[:10]}"
        if locality: body += f" near {locality}"
        body += ". Want one practical operating adjustment to consider?"
        return {"score": score + 20, "body": body, "cta": "open_ended", "facts": [event, str(p.get("impact_window") or ""), str(p.get("date") or "")], "label": kind}
    if kind in {"festival_upcoming", "category_seasonal", "weather_heatwave", "local_news_event", "dormant_with_vera", "winback_eligible", "gbp_unverified", "scheduled_recurring"}:
        facts = [str(v) for v in p.values() if isinstance(v, (str, int, float)) and v is not None][:3]
        subject = str(p.get("festival") or p.get("season") or p.get("event") or p.get("topic") or kind.replace("_", " "))
        body = f"{who}, a timely {subject} update may be relevant to {biz}"
        if locality: body += f" in {locality}"
        if facts: body += ". Context: " + ", ".join(facts[:2])
        body += ". Want me to suggest one category-fit next step?"
        return {"score": score + 20, "body": body, "cta": "open_ended", "facts": facts, "label": kind}
    # Unknown kinds may be messaged only when a small, safe fact is present.
    safe_keys = ("title", "topic", "metric", "value", "date", "deadline", "event", "summary", "source", "name")
    safe_facts = [str(p[k]).strip() for k in safe_keys if isinstance(p.get(k), (str, int, float)) and str(p[k]).strip()]
    if not safe_facts:
        return {"score": -1, "body": "", "cta": "none", "facts": [], "label": kind}
    subject = ", ".join(safe_facts[:2])
    body = f"Hi {who}, {biz}: {subject}. Want a short summary of this update?"
    return {"score": score, "body": body, "cta": "open_ended", "facts": safe_facts[:2], "label": kind}


def _decision_score(category: dict, merchant: dict, trigger: dict, customer: dict | None, signal: dict) -> int:
    """Rank eligible candidates from evidence and fit; urgency is one feature, not the decision."""
    payload = trigger.get("payload") or {}
    cat = _category(merchant)
    facts = [str(x).strip() for x in signal.get("facts", []) if str(x).strip()]
    score = _urgency(trigger) * 4
    score += min(20, len(set(facts)) * 4)

    timely_fields = {
        "date", "due_date", "deadline_iso", "days_until", "days_remaining", "stock_runs_out_iso",
        "match_time_iso", "appointment_date", "opened_date", "expires_at", "window",
    }
    if any(payload.get(field) not in (None, "", []) for field in timely_fields):
        score += 8
    elif trigger.get("kind") in {"research_digest", "supply_alert", "regulation_change", "cde_opportunity"} and facts:
        score += 6

    known_verticals = {"dentists", "salons", "restaurants", "gyms", "pharmacies"}
    if cat in known_verticals and category:
        score += 5
    identity = merchant.get("identity") or {}
    if identity.get("name") or identity.get("owner_first_name"):
        score += 3
    if identity.get("locality") and cat in {"restaurants", "salons", "gyms"}:
        score += 2

    # A trigger gains fit when it connects to the actual performance or offer context.
    perf = merchant.get("performance") or {}
    metric = str(payload.get("metric") or "").lower()
    if metric and metric in {str(k).lower() for k in perf}:
        score += 5
    signals = merchant.get("signals") or []
    signal_text = " ".join(str(x).lower() for x in signals)
    kind_words = str(trigger.get("kind") or "").replace("_", " ").lower()
    if signal_text and any(word in signal_text for word in kind_words.split() if len(word) > 3):
        score += 4
    if merchant.get("offers") and any(o.get("status", "active") == "active" for o in merchant.get("offers", []) if isinstance(o, dict)):
        if any(word in str(trigger.get("kind") or "").lower() for word in ("offer", "match", "recall", "winback", "festival")):
            score += 4

    if customer:
        relationship = customer.get("relationship") or {}
        prefs = customer.get("preferences") or {}
        if relationship.get("last_visit") or relationship.get("services_received"):
            score += 4
        if prefs.get("preferred_time") or _language_preference(customer):
            score += 2
    if signal.get("cta") not in {None, "none"}:
        score += 2
    return score


def _compose_ranked(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> tuple[dict[str, Any], int]:
    if trigger.get("scope") == "customer" and (
        customer is None
        or str(trigger.get("kind") or "") not in CUSTOMER_TRIGGER_KINDS
        or customer.get("merchant_id") != merchant.get("merchant_id")
    ):
        selected = {"score": -1, "body": "", "cta": "none", "facts": [], "label": str(trigger.get("kind") or "unknown")}
    else:
        selected = _signal(trigger, category or {}, merchant or {}, customer)
    if selected.get("body") and trigger.get("scope") != "customer":
        selected["body"] = _localized_cta(selected["body"], merchant, audience="merchant")
    score = _decision_score(category or {}, merchant or {}, trigger, customer, selected) if selected["body"] else -1
    key = str(trigger.get("suppression_key") or f"{merchant.get('merchant_id', '')}:{trigger.get('kind', '')}:{trigger.get('id', '')}")
    output = {
        "body": selected["body"], "cta": selected["cta"],
        "send_as": selected.get("send_as", "vera" if not customer else "merchant_on_behalf"),
        "suppression_key": key,
        "rationale": (f"Selected {selected['label']} using urgency, grounded facts, timing, and merchant/context fit (score {score}); message facts come from the latest pushed contexts." if selected["body"] else f"No worthwhile eligible message for {selected['label']}: insufficient grounded facts or required context."),
    }
    return output, score


def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict[str, Any]:
    """Compose only the documented message fields for local evaluation."""
    return _compose_ranked(category, merchant, trigger, customer)[0]


def _valid_customer(trigger: dict, merchant_id: str) -> dict | None:
    cid = trigger.get("customer_id")
    customer = _context("customer", str(cid)) if cid else None
    if not customer or customer.get("merchant_id") != merchant_id:
        return None
    return customer


@app.middleware("http")
async def payload_limit(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/v1/context":
        body = await request.body()
        if len(body) > MAX_CONTEXT_BYTES:
            return JSONResponse(status_code=413, content={"accepted": False, "reason": "payload_too_large", "details": "Context payload exceeds 500 KB"})
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    details = "; ".join(f"{'.'.join(str(x) for x in err.get('loc', []))}: {err.get('msg', 'invalid value')}" for err in exc.errors())
    return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_request", "details": details})


@app.get("/v1/healthz")
def healthz():
    counts = {s: 0 for s in ("category", "merchant", "customer", "trigger")}
    with lock:
        for scope, _ in contexts:
            counts[scope] += 1
    return {"status": "ok", "uptime_seconds": int(time.monotonic() - STARTED), "contexts_loaded": counts}


@app.get("/v1/metadata")
def metadata():
    return {"team_name": "Vera Deterministic Systems", "team_members": [], "model": "deterministic-python", "approach": "Grounded signal ranking with category-aware templates and in-memory versioned state", "contact_email": "", "version": "1.0.0", "submitted_at": now_iso()}


@app.post("/v1/context")
def push_context(body: ContextPush):
    if not body.payload:
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_payload", "details": "payload must be a non-empty JSON object"})
    key = (body.scope, body.context_id)
    with lock:
        cur = contexts.get(key)
        if cur and body.version == cur["version"] and body.payload == cur["payload"]:
            return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": now_iso()}
        if cur and body.version <= cur["version"]:
            return JSONResponse(status_code=409, content={"accepted": False, "reason": "stale_version", "current_version": cur["version"]})
        contexts[key] = {"version": body.version, "payload": body.payload, "delivered_at": body.delivered_at}
    log.info("context stored scope=%s id=%s version=%d", body.scope, body.context_id, body.version)
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": now_iso()}


@app.post("/v1/tick")
def tick(body: TickRequest):
    actions = []
    candidates = []
    tick_time = _parse_time(body.now)
    with lock:
        for trigger_id in body.available_triggers:
            trigger = _context("trigger", trigger_id)
            if not trigger:
                continue
            mid = str(trigger.get("merchant_id") or "")
            merchant = _context("merchant", mid)
            if not merchant:
                continue
            cat_slug = str(merchant.get("category_slug") or "")
            category = _context("category", cat_slug)
            if not category:
                continue
            customer = _valid_customer(trigger, mid) if trigger.get("scope") == "customer" else None
            if trigger.get("scope") == "customer" and customer is None:
                continue
            if trigger.get("scope") == "customer" and str(trigger.get("kind") or "") not in CUSTOMER_TRIGGER_KINDS:
                continue
            expires_at = _parse_time(trigger.get("expires_at"))
            if tick_time and expires_at and expires_at <= tick_time:
                continue
            result, score = _compose_ranked(category, merchant, trigger, customer)
            suppression = result["suppression_key"]
            customer_id = str(trigger.get("customer_id") or "")
            content_fingerprint = hashlib.sha256(result["body"].encode("utf-8")).hexdigest()[:10]
            internal_suppression = f"{mid}:{customer_id or 'merchant'}:{suppression}:{content_fingerprint}"
            content_key = f"{mid}:{customer_id or 'merchant'}:{content_fingerprint}"
            if (
                internal_suppression in sent_suppression
                or content_key in sent_content
                or (mid, customer_id) in blocked_contacts
                or not result["body"]
                or score < 0
            ):
                continue
            # The harness's available_triggers list is authoritative for the simulated window;
            # it can intentionally use historical seed timestamps.
            candidates.append((score, trigger_id, mid, trigger, customer, result, suppression, internal_suppression))
        # Rank by explicit signal score, then urgency and stable trigger ID.
        candidates.sort(key=lambda c: (-c[0], -int(c[3].get("urgency") or 1), c[1]))
        chosen_merchants: set[str] = set()
        for _, trigger_id, mid, trigger, customer, result, suppression, internal_suppression in candidates:
            if mid in chosen_merchants:
                continue
            conv_id = f"conv_{mid}_{trigger_id}"
            if conv_id in conversations:
                conv_fingerprint = hashlib.sha256(result["body"].encode("utf-8")).hexdigest()[:10]
                conv_id = f"{conv_id}_{conv_fingerprint}"
            if conv_id in conversations:
                continue
            template_name = f"vera_{trigger.get('kind', 'update')}_v1"
            actions.append({
                "conversation_id": conv_id, "merchant_id": mid,
                "customer_id": trigger.get("customer_id") if customer else None,
                "send_as": result["send_as"], "trigger_id": trigger_id,
                "template_name": template_name,
                "template_params": [result["body"]], "body": result["body"],
                "cta": result["cta"], "suppression_key": suppression,
                "rationale": result["rationale"],
            })
            conversations[conv_id] = {"merchant_id": mid, "customer_id": trigger.get("customer_id"), "trigger_id": trigger_id, "body": result["body"], "turns": [], "ended": False, "auto_replies": 0, "suppression": suppression}
            sent_suppression.add(internal_suppression)
            sent_content.add(content_key)
            chosen_merchants.add(mid)
            if len(actions) >= MAX_ACTIONS:
                break
    return {"actions": actions}


AUTO_REPLY = ("thank you for contacting", "our team will respond", "automated assistant", "for your information", "aapki jaankari ke liye", "team tak pahunchati")
STOP = ("stop messaging", "unsubscribe", "not interested", "don't message", "do not message", "do not contact", "remove me", "useless spam")
YES = ("yes", "ok", "okay", "lets do it", "let's do it", "go ahead", "interested", "please do", "send it", "sure")
LATER = ("later", "not now", "remind me", "tomorrow", "i'm busy", "im busy", "i am busy", "busy right now")


def _grounded_reply_detail(state: dict, message: str) -> str | None:
    trigger = _context("trigger", str(state.get("trigger_id") or ""))
    if not trigger:
        return None
    merchant = _context("merchant", str(state.get("merchant_id") or "")) or {}
    category = _context("category", str(merchant.get("category_slug") or "")) or {}
    payload = trigger.get("payload") or {}
    customer = _context("customer", str(state.get("customer_id") or "")) if state.get("customer_id") else None
    lowered = message.lower()
    if any(term in lowered for term in ("price", "cost", "offer")):
        offer = _active_offer(merchant)
        if offer:
            return f"The active offer currently listed is {offer}. I can use that verified detail in the draft."
        price = payload.get("price") or payload.get("amount") or payload.get("fee")
        if price is not None:
            return f"The trigger context lists {price} as the amount. I can use that detail in the draft."
        return "I don't have a current price in the pushed context, so I won't guess. Share the verified price and I can include it."
    if trigger.get("kind") in {"research_digest", "regulation_change", "cde_opportunity", "supply_alert"}:
        item = _digest_item(category, trigger)
        if item:
            summary = str(item.get("summary") or item.get("title") or "").strip()
            source = str(item.get("source") or "").strip()
            if summary and source:
                return f"The pushed note says: {summary} Source: {source}. I can turn that into one practical next step."
            if summary:
                return f"The pushed note says: {summary}. I can turn that into one practical next step."
    for field, label in (("deadline_iso", "deadline"), ("due_date", "due date"), ("date", "date"), ("metric", "metric"), ("window", "window"), ("days_remaining", "days remaining")):
        if payload.get(field) not in (None, "") and any(term in lowered for term in ("when", "what", "which", "detail", "date", "deadline", "metric")):
            return f"The latest trigger context lists {label}: {payload[field]}. I can use that exact detail in the next draft."
    if customer:
        slots = payload.get("available_slots") or []
        labels = _slot_labels(slots, customer)
        if labels and any(term in lowered for term in ("when", "time", "slot", "available")):
            return f"The context lists {', '.join(labels)}. Which one should the merchant follow up about?"
    return None


@app.post("/v1/reply")
def reply(body: ReplyRequest):
    msg = re.sub(r"\s+", " ", body.message.strip().lower())
    with lock:
        state = conversations.setdefault(body.conversation_id, {"merchant_id": body.merchant_id, "customer_id": body.customer_id, "turns": [], "ended": False, "auto_replies": 0})
        if state.get("merchant_id") and body.merchant_id and str(state["merchant_id"]) != body.merchant_id:
            return {"action": "end", "rationale": "The reply does not match this conversation's merchant."}
        if state.get("customer_id") and body.customer_id and str(state["customer_id"]) != body.customer_id:
            return {"action": "end", "rationale": "The reply does not match this conversation's customer."}
        state.setdefault("merchant_id", body.merchant_id)
        state.setdefault("customer_id", body.customer_id)
        state.setdefault("turns", []).append({"role": body.from_role, "message": body.message, "received_at": body.received_at, "turn_number": body.turn_number})
        if state.get("ended"):
            return {"action": "end", "rationale": "This conversation is already closed."}
        normalized_stop = re.sub(r"[.!?]+$", "", msg).strip()
        if normalized_stop == "stop" or any(_contains_phrase(msg, x) for x in STOP):
            state["ended"] = True
            blocked_contacts.add((str(body.merchant_id or state.get("merchant_id") or "unknown"), str(body.customer_id or state.get("customer_id") or "")))
            return {"action": "end", "rationale": "The recipient asked to stop; closing the conversation."}
        if any(_contains_phrase(msg, x) for x in AUTO_REPLY):
            state["auto_replies"] = state.get("auto_replies", 0) + 1
            auto_key = (str(body.merchant_id or state.get("merchant_id") or "unknown"), msg)
            auto_reply_counts[auto_key] = auto_reply_counts.get(auto_key, 0) + 1
            if state["auto_replies"] >= 2 or auto_reply_counts[auto_key] >= 2:
                state["ended"] = True
                return {"action": "end", "rationale": "Repeated canned auto-reply detected; ending to avoid wasting turns."}
            return {"action": "wait", "wait_seconds": 14400, "rationale": "Canned business auto-reply detected; waiting for the owner instead of prompting again."}
        if any(_contains_phrase(msg, x) for x in LATER):
            return {"action": "wait", "wait_seconds": 1800, "rationale": "Recipient asked for a later follow-up; backing off for 30 minutes."}
        # Avoid substring positives such as "yesterday" or negative phrases such as "not sure".
        negative = any(_contains_phrase(msg, x) for x in ("no", "not yet", "not sure", "maybe later", "don't", "do not"))
        affirmative = any(_contains_phrase(msg, x) for x in YES)
        if affirmative and not negative:
            if body.from_role == "customer":
                state["intent_confirmed"] = True
                return {"action": "send", "body": "Thanks — the business will follow up with the details shortly.", "cta": "none", "rationale": "Acknowledged the customer's confirmation and avoided inventing availability or fulfillment details."}
            state["intent_confirmed"] = True
            return {"action": "send", "body": "Great, I'll prepare the next step from the details already shared and bring it back here for your review.", "cta": "none", "rationale": "Recognized clear intent and moved to execution without another qualification question."}
        if any(_contains_phrase(msg, x) for x in ("gst", "tax filing", "unrelated", "off topic", "off-topic")):
            return {"action": "send", "body": "I can’t help with tax filing, but your CA can. I can still help with the business-growth update we were discussing.", "cta": "open_ended", "rationale": "Politely declined an out-of-scope request and redirected to the active topic."}
        if "?" in msg or any(_contains_phrase(msg, x) for x in ("how", "what", "price", "details", "more", "when", "why", "which")):
            detail = _grounded_reply_detail(state, msg)
            return {"action": "send", "body": detail or "I can clarify the trigger detail I have. Which single point should I explain first?", "cta": "open_ended", "rationale": "Answered from the latest available context when possible, without inventing missing details."}
        return {"action": "send", "body": "Understood. I can keep this focused on the update we discussed; would a short summary help?", "cta": "open_ended", "rationale": "Acknowledged the reply and offered a low-effort continuation without repeating the original message."}


@app.post("/v1/teardown")
def teardown():
    with lock:
        contexts.clear()
        conversations.clear()
        sent_suppression.clear()
        sent_content.clear()
        auto_reply_counts.clear()
        blocked_contacts.clear()
    return {"accepted": True, "contexts_cleared": True}
