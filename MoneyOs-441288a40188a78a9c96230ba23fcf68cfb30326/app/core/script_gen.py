import random
import re
from dataclasses import dataclass

from app.config import MIN_AUDIO_SECONDS

WORDS_PER_SECOND = 2.8
TARGET_MIN_WORDS = 1300
TARGET_MAX_WORDS = 1600


@dataclass
class ScriptResult:
    text: str
    estimated_seconds: float


def _estimate_seconds(text: str) -> float:
    word_count = len(text.split())
    return word_count / WORDS_PER_SECOND


def sanitize_script(text: str) -> str:
    cleaned = re.sub(r"[#\*_{}\[\]|><]", " ", text)
    cleaned = re.sub(r"`{1,3}.*?`{1,3}", " ", cleaned, flags=re.DOTALL)
    lines = [re.sub(r"\s+", " ", line).strip() for line in cleaned.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _starter_pool() -> list[str]:
    return [
        "First",
        "Next",
        "Then",
        "Afterward",
        "Meanwhile",
        "Suddenly",
        "Oddly",
        "Quietly",
        "Loudly",
        "Inside",
        "Outside",
        "Across",
        "Behind",
        "Beneath",
        "Above",
        "Beyond",
        "Before",
        "Later",
        "Earlier",
        "Eventually",
        "Briefly",
        "Frankly",
        "Naturally",
        "Carefully",
        "Boldly",
        "Softly",
        "Slowly",
        "Quickly",
        "Cautiously",
        "Firmly",
        "Gently",
        "Sharply",
        "Still",
        "Also",
        "Instead",
        "Rather",
        "Otherwise",
        "Because",
        "Since",
        "Although",
        "While",
        "Unless",
        "Even",
        "Yet",
        "So",
        "Therefore",
        "However",
        "Nevertheless",
        "Moreover",
        "Furthermore",
        "Consequently",
        "Similarly",
        "Likewise",
        "Specifically",
        "Practically",
        "Notably",
        "Remarkably",
        "Surprisingly",
        "Predictably",
        "Unusually",
        "Strangely",
        "Decisively",
        "Precisely",
        "Ultimately",
        "Reluctantly",
        "Patiently",
        "Tensely",
        "Calmly",
        "Uneasily",
        "Deliberately",
        "Publicly",
        "Privately",
        "Silently",
        "Openly",
        "Directly",
        "Indirectly",
        "Technically",
        "Measured",
        "Steadily",
        "Straightaway",
        "Immediately",
        "Right",
        "Left",
        "Up",
        "Down",
        "Forward",
        "Back",
        "North",
        "South",
        "East",
        "West",
        "Today",
        "Tonight",
        "Tomorrow",
        "Yesterday",
        "Always",
        "Never",
        "Finally",
        "Last",
        "Already",
        "Again",
        "Once",
        "Twice",
        "Stillness",
        "Truthfully",
        "Honestly",
        "Plainly",
        "Simply",
        "Seriously",
        "Careful",
        "Focused",
        "Determined",
        "Balanced",
        "Grounded",
        "Measuredly",
        "Plainspoken",
        "Levelly",
        "Exactingly",
    ]


def _sentence(starter: str, body: str) -> str:
    sentence = f"{starter} {body}".strip()
    if not sentence.endswith((".", "!", "?")):
        sentence = f"{sentence}."
    return sentence


def _apply_starters(bodies: list[str], starters: list[str]) -> list[str]:
    sentences = []
    for body in bodies:
        if not starters:
            starters.extend(_starter_pool())
        starter = starters.pop(0)
        sentences.append(_sentence(starter, body))
    return sentences


def _section(title: str, bodies: list[str], starters: list[str]) -> list[str]:
    return [title, *_apply_starters(bodies, starters), ""]


def generate_script(min_seconds: int = MIN_AUDIO_SECONDS) -> ScriptResult:
    protagonist = "Mara"
    ally = "Eli"
    town = "Ridgeview"
    archive = "the riverside archive"
    office = "the old co-op office"

    starters = _starter_pool()
    lines: list[str] = []

    hook_bodies = [
        f"I can promise you one answer: who moved {town}'s emergency fund and why.",
        "We start with a missing ledger and a call that cut off mid-sentence.",
        "Stay with me, because the truth flips the story you expect.",
    ]
    context_bodies = [
        f"{protagonist} handled budget notes for the relief fund, a job that normally never made headlines.",
        f"{ally} managed the daily requests and knew which families were barely holding on.",
        f"The fund was meant for storms, layoffs, and the quiet emergencies no one wants to name.",
        f"In {town}, the money sat in a reserve account with strict rules and simple signatures.",
        f"The last clean record lived in {archive}, filed under a schedule that never drifted.",
        f"On paper, the account balance was steady the week before the shock.",
        f"The next morning, the account read almost empty with no public notice.",
        f"{protagonist} and {ally} decided to trace the change before rumors did it for them.",
    ]
    escalation_one_bodies = [
        f"The first clue was a timestamp that showed the transfer happened well after midnight.",
        f"Security logs from {office} showed a keycard entry that should not exist.",
        f"A vendor invoice appeared with the same amount as the missing reserve.",
        f"{ally} called the bank and learned the transfer had been approved twice.",
        f"One approval came from a name that no longer worked there.",
        f"{protagonist} pulled the paper trail and saw that the ledger was edited, not erased.",
        f"A backup folder had a single file renamed with the wrong date.",
        f"The finance chair insisted the board never met that night.",
        f"An email chain showed a meeting invite that no one remembered accepting.",
        f"A short voice note hinted the reserve was being moved for “temporary protection.”",
        f"A note in the margin said to wait until the audit window closed.",
        f"{ally} realized the audit window closed the same day the fund vanished.",
        f"{protagonist} checked the stamp on the ledger and found it matched a batch from months earlier.",
        f"A quiet clerk admitted the stamp box had been taken home once.",
        f"The trail now pointed to intent, not a mistake.",
    ]
    escalation_two_bodies = [
        "The complication was that the transfer did not go to a private account.",
        "The money moved into a legal escrow tied to a redevelopment bid.",
        f"That bid would decide whether {town} kept control of its emergency services.",
        f"{ally} worried that the escrow meant the fund could be frozen for months.",
        f"{protagonist} found a letter suggesting the fund would be seized if left untouched.",
        "A retired treasurer warned that the bank had a clause nobody had read in years.",
        f"The clause required the reserve to remain above a threshold during the bid.",
        "Yet the transfer had lowered the visible balance below that line.",
        "The contradiction was brutal: the move both protected and endangered the town.",
        f"Pressure rose as families asked why emergency checks had paused.",
        f"{ally} faced those questions directly, while {protagonist} kept digging.",
        "Every explanation sounded like a cover story, and none solved the timing.",
        f"A second ledger copy showed a prepared statement labeled “if discovered.”",
        "The statement claimed a hacker forced the transfer, but the logs showed no breach.",
        f"By this point, {protagonist} knew someone inside made the call.",
    ]
    turn_bodies = [
        f"The reveal came in {archive} when a sealed file finally opened.",
        "It contained a legal warning about a predatory lawsuit that would drain the fund in days.",
        f"{protagonist} saw the signature and realized the transfer was an emergency shield, not a theft.",
        f"{ally} had been kept out to avoid exposing the plan too early.",
        f"The mystery shifted from “who stole it” to “who risked everything to save it.”",
    ]
    payoff_bodies = [
        f"The answer was clear: {protagonist} moved the fund into escrow to block the lawsuit from touching it.",
        "The late-night timing was chosen because the injunction clock started at dawn.",
        "The double approval was a workaround for a board that could not meet in time.",
        f"The fake cover story was written to keep {town} calm until the legal window passed.",
        f"The escrow terms required silence, which is why even {ally} was left in the dark.",
        "Once the lawsuit was dismissed, the reserve could return intact.",
        "That is why the balance appeared empty while the money was actually protected.",
        "No corruption was found, only a risky maneuver to keep services alive.",
        f"The fund returned in full, and the families in {town} received the aid they needed.",
        "The promise in the hook is kept: the money never vanished, it was shielded.",
    ]
    landing_bodies = [
        f"The final insight is that trust survives when people explain their risks before panic fills the gap.",
        f"{ally} forgave the secrecy because the outcome saved the community.",
        "A single question can feel like a scandal until the full context arrives.",
        "If you want more stories where the answer changes how you see the whole chain, stay curious.",
    ]

    sections = [
        ("SECTION 1 — HOOK", hook_bodies),
        ("SECTION 2 — CONTEXT SETUP", context_bodies),
        ("SECTION 3 — ESCALATION PHASE 1", escalation_one_bodies),
        ("SECTION 4 — ESCALATION PHASE 2", escalation_two_bodies),
        ("SECTION 5 — TURN / REFRAME", turn_bodies),
        ("SECTION 6 — PAYOFF", payoff_bodies),
        ("SECTION 7 — LANDING", landing_bodies),
    ]

    for title, bodies in sections:
        lines.extend(_section(title, bodies, starters))

    script = "\n".join(lines).strip()
    word_count = len(script.split())
    padding_bodies = [
        "The bank statements showed the transfer as a protective hold, not a withdrawal.",
        "A timeline on the wall proved how quickly the legal window was closing.",
        "The relief team had already prepared contingency plans for a short delay.",
        "Public minutes from the last council meeting hinted at the lawsuit without naming it.",
        "Every document pointed to urgency rather than greed.",
        "The more they verified, the more the plan looked intentional and time-bound.",
        "Small details like courier receipts confirmed the escrow paperwork was filed on time.",
        "Even the auditor admitted the decision followed the letter of the rules.",
        "By sunrise, the danger had shifted from theft to the risk of misunderstanding.",
        "That misunderstanding was the real threat to trust in the fund.",
    ]

    while word_count < TARGET_MIN_WORDS:
        if not padding_bodies:
            padding_bodies.append("The choice was risky, but the alternative was losing everything.")
        extra = _apply_starters([padding_bodies.pop(0)], starters)
        lines.insert(-1, extra[0])
        script = "\n".join(lines).strip()
        word_count = len(script.split())

    if word_count > TARGET_MAX_WORDS:
        trimmed_lines = [line for line in lines if line]
        while trimmed_lines and len(" ".join(trimmed_lines).split()) > TARGET_MAX_WORDS:
            trimmed_lines.pop(-1)
        script = "\n".join(trimmed_lines).strip()

    script = sanitize_script(script)
    return ScriptResult(text=script, estimated_seconds=_estimate_seconds(script))


def expand_script_once(script_text: str) -> ScriptResult:
    return ScriptResult(text=script_text, estimated_seconds=_estimate_seconds(script_text))


def generate_titles(script_text: str) -> list[str]:
    titles = [
        "The Small Moment That Changed Everything",
        "A Story About Trust, Pressure, and One Wrong Move",
        "The Night the Plan Slipped Away",
        "How a Quiet Promise Became a Mess",
        "The Story No One Wanted to Tell",
        "What Really Happened After the Warning",
    ]
    random.shuffle(titles)
    return titles[:3]


def generate_description(script_text: str) -> str:
    return (
        "A calm, conversational story with a clear beginning, rising tension, "
        "a reveal, and a grounded ending, told like a real person walking you "
        "through what happened and why it mattered."
    )
