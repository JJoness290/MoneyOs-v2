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

def generate_script(min_seconds: int = MIN_AUDIO_SECONDS) -> ScriptResult:
    protagonist = "Mara"
    ally = "Eli"
    town = "Ridgeview"

    hook = [
        f"I promised one clear answer about who moved {town}'s emergency fund and why it happened.",
        "The story begins with a missing ledger and a phone call that ended mid-sentence.",
        "That promise will be kept, and the answer changes how the whole chain looks.",
        "No room for guesses.",
    ]
    context = [
        f"{protagonist} kept the relief fund records and signed off on routine checks.",
        f"{ally} tracked the requests that came in each day and saw the pressure building.",
        "This fund existed for storms, layoffs, and the quiet emergencies that never reach the news.",
        f"{town} stored it in a reserve account with simple signatures and a small audit window.",
        "A clean report showed a steady balance late in the week.",
        "The next morning brought an almost empty balance and no public note to explain it.",
        f"{protagonist} and {ally} agreed to trace the trail before rumors hardened into fact.",
    ]
    escalation_one = [
        "A time stamp showed the transfer happened after midnight.",
        "A keycard log showed entry from a card that should have been inactive.",
        "A vendor invoice matched the missing amount down to the dollar.",
        f"{ally} called the bank and learned the transfer had two approvals.",
        "One approval belonged to a name removed from the staff list.",
        f"{protagonist} pulled a paper copy and saw edits instead of a full erase.",
        "A backup folder carried a file with the wrong date in its title.",
        "A finance chair swore the board never met that night.",
        "An old invite showed a meeting time that nobody recalled.",
        "A voice note hinted the reserve was being moved for protection.",
        "A margin note said to wait until the audit window closed.",
        f"{ally} saw the audit window ended the same day the fund vanished.",
        f"{protagonist} checked the stamp and found it matched a batch from months earlier.",
        "A clerk admitted the stamp box had gone home with someone once.",
        "A transfer memo listed a routing code that did not match prior months.",
        f"{protagonist} compared the routing code to old files and found a mismatch.",
        f"{ally} noticed the approval times were clustered within five minutes.",
        "One signature line used a block stamp instead of pen.",
        "The stamp ink matched a box stored near the archives.",
        "A quiet assistant confirmed the box went missing for a weekend.",
        "The weekend aligned with the unexplained meeting invite.",
        "A bank clerk confirmed the escrow form was prefilled before midnight.",
        "A draft email showed a warning about a looming injunction.",
        "The warning made the midnight transfer feel like a deadline.",
        "That trail pointed to intent rather than a mistake.",
    ]
    escalation_two = [
        "The transfer did not land in a private account.",
        "That money moved into legal escrow tied to a redevelopment bid.",
        f"That bid would decide whether {town} kept control of its emergency services.",
        f"{ally} worried the escrow could freeze the fund for months.",
        f"{protagonist} found a letter warning the fund could be seized if left exposed.",
        "A retired treasurer mentioned a clause buried in the bank agreement.",
        "That clause required the reserve to stay above a threshold during the bid.",
        "The transfer lowered the visible balance below that line.",
        "That risk felt immediate.",
        "That contradiction felt sharp: the move protected the cash and endangered it.",
        f"Families asked why emergency checks paused, and {ally} had no answer.",
        f"{protagonist} kept digging while rumors circled back to blame.",
        "Each explanation sounded like a cover story and none solved the timing.",
        "By this point, one thing was clear: the transfer was deliberate, and the clock was forcing hard choices.",
        "A second ledger copy carried a prepared statement labeled for discovery.",
        "The statement blamed a hacker, yet the logs showed no breach.",
        f"{protagonist} concluded the choice came from inside.",
        "A legal aide described the escrow as a temporary safe room.",
        "The note explained that the reserve could not be garnished there.",
        f"{protagonist} realized the escrow would look like a loss on casual review.",
        f"{ally} feared that review would spark panic before the hold lifted.",
    ]
    pressure_details = [
        "Bank statements showed a protective hold instead of a withdrawal.",
        "A calendar on the wall proved how small the legal window was.",
        "Every hour felt tighter.",
        "The relief team prepared for a short delay while the escrow closed.",
        "Public minutes hinted at legal pressure without naming it.",
        "Every document pointed to urgency rather than greed.",
        "Courier receipts confirmed the escrow paperwork was filed on time.",
        "An auditor admitted the decision followed the letter of the rules.",
        "The risk now was misunderstanding rather than missing money.",
        "The public needed clarity more than a villain.",
        "Meeting notes showed the deadline driving every move.",
        "A stack of receipts tracked each step through the transfer.",
        "The bank officer stressed that the escrow held every dollar.",
        "The relief staff documented every call they could not fulfill.",
        "A quiet promise to reopen the fund held the team together.",
        "Paper copies showed the fund never left the system.",
        "A timeline on the whiteboard narrowed the window of danger.",
        "The reserve could only be touched once the lawsuit threat expired.",
        f"{protagonist} reviewed every signature to confirm the chain held firm.",
        f"{ally} watched the requests pile up and felt the strain grow.",
        "A simple checklist kept the records consistent under stress.",
        "The bank agreement treated the escrow as a shield, not a spend.",
        "Each verified detail reduced the fear of a hidden theft.",
        "The reserve stayed safe even while the balance looked wrong.",
        "The phone logs showed the urgency behind the late-night call.",
        "A clean audit trail remained intact despite the panic.",
        "The oversight file listed the legal risks in plain language.",
        "The decision aimed to keep emergency services running.",
        f"{protagonist} kept a printed copy of every notice for backup.",
        f"{ally} kept the team calm with steady updates.",
        "A brief pause in aid felt painful but temporary.",
        "The town council expected a short-term freeze and prepared notices.",
        "A handwritten note explained the escrow rule in simple terms.",
        "The escrow clause described the reserve as protected property.",
        "The bank manager confirmed the funds could not be seized while held.",
        "A legal summary tied the transfer to a single filing deadline.",
        "The finance chair admitted the clock left no easy option.",
        "A clear timeline helped the team see the logic of the move.",
        "The staff kept receipts ready for the audit review.",
        "The records room held duplicate logs for verification.",
        "The reserve stayed intact while the lawsuit threat expired.",
        "A written plan outlined the steps to restore the balance.",
        "The team prepared a public statement for the moment the hold lifted.",
        "A quiet sense of relief arrived as the legal pressure eased.",
        "The last check confirmed the escrow release had been scheduled.",
        "The relief team agreed to rebuild trust through transparency.",
    ]
    escalation_two.extend(pressure_details)
    turn = [
        "A sealed file in the archive finally opened after a records request.",
        "The file contained a legal warning about a predatory lawsuit due within days.",
        f"{protagonist} read the signature and saw the transfer as a shield, not a theft.",
        f"{ally} had been kept out to avoid exposing the plan too soon.",
        "The mystery shifted from theft to the risk taken to save the fund.",
    ]
    payoff = [
        f"The answer is direct: {protagonist} moved the fund into escrow to block the lawsuit.",
        "The late-night timing mattered because the injunction clock started at dawn.",
        "The double approval was a legal workaround for a board that could not meet fast enough.",
        f"The cover story kept {town} calm until the legal window passed.",
        f"The escrow terms demanded silence, which is why even {ally} was left in the dark.",
        "The lawsuit was dismissed, and the reserve returned intact.",
        "The balance looked empty while the money stayed protected.",
        "No corruption appeared, only a risky maneuver to keep services alive.",
        f"The reserve returned in full, and the families in {town} received the aid they needed.",
        "The promise is kept: the money never vanished, it was shielded.",
    ]
    landing = [
        "Trust survives when people explain the risk before panic fills the gap and before rumors harden into blame.",
        f"{ally} carried the fear of failure, felt the weight of every request, and still chose to forgive the secrecy.",
        "The relief in that decision mattered as much as the money because it rebuilt the team itself.",
        "The story proves how quickly a protective act can look like betrayal when the timeline is hidden.",
        "A single answer can change what a whole chain of events means and how a community heals.",
        "Let that sit if you have ever doubted a quiet decision made under pressure.",
        "Stay curious if you want more stories where one truth reshapes everything.",
    ]

    sentences = [*hook, "", *context, "", *escalation_one, "", *escalation_two, "", *turn, "", *payoff, ""]
    landing_text = " ".join(landing)

    script = " ".join(sentences).strip()
    script = f"{script} {landing_text}".strip()
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
