import random
import re
from dataclasses import dataclass

from app.config import MIN_AUDIO_SECONDS

WORDS_PER_SECOND = 3.0


@dataclass
class ScriptResult:
    text: str
    estimated_seconds: float


def _estimate_seconds(text: str) -> float:
    word_count = len(text.split())
    return word_count / WORDS_PER_SECOND


def sanitize_script(text: str) -> str:
    cleaned = re.sub(r"[#/\*_{}\[\]|><]", " ", text)
    cleaned = re.sub(r"`{1,3}.*?`{1,3}", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _sentence_similarity(candidate: str, existing: str) -> float:
    candidate_tokens = set(candidate.lower().split())
    existing_tokens = set(existing.lower().split())
    if not candidate_tokens or not existing_tokens:
        return 0.0
    overlap = candidate_tokens.intersection(existing_tokens)
    return len(overlap) / max(len(candidate_tokens), len(existing_tokens))


def _is_repetitive(candidate: str, used: list[str]) -> bool:
    return any(_sentence_similarity(candidate, prior) >= 0.6 for prior in used)


def _pick_unique(options: list[str], used: set[str]) -> str:
    choices = [option for option in options if option not in used]
    if not choices:
        return random.choice(options)
    choice = random.choice(choices)
    used.add(choice)
    return choice


def _name_pool() -> list[str]:
    return [
        "Mara",
        "Eli",
        "Jonah",
        "Priya",
        "Tess",
        "Owen",
        "Nina",
        "Caleb",
        "Rue",
        "Samir",
        "Lena",
        "Jo",
        "Kian",
        "Milo",
        "Anya",
        "Inez",
        "Mateo",
    ]


def _location_pool() -> list[str]:
    return [
        "the quiet riverfront",
        "a sun-baked bus depot",
        "the worn lobby of a tech co-op",
        "a closed diner with flickering neon",
        "the third floor of a public library",
        "a crowded night market",
        "an old workshop behind a corner store",
        "the rooftop of a parking garage",
        "a narrow alley lined with murals",
        "the back room of a community center",
    ]


def _object_pool() -> list[str]:
    return [
        "a ledger with torn pages",
        "a dented phone",
        "a thrifted backpack",
        "a keycard with a smudged logo",
        "a crumpled map",
        "a chipped mug",
        "a photo booth strip",
        "a folded note",
        "a scratched flash drive",
        "a taped-up badge",
    ]


def _traits_pool() -> list[str]:
    return [
        "restless",
        "careful",
        "soft-spoken",
        "blunt",
        "curious",
        "skeptical",
        "warm",
        "stubborn",
        "observant",
        "tired but kind",
    ]


def _beat_types() -> list[str]:
    return [
        "setup",
        "inciting",
        "complication",
        "pressure",
        "reveal",
        "consequence",
        "turn",
        "resolution",
        "aftermath",
    ]


def _short_sentences() -> list[str]:
    return [
        "{name} hesitated when {detail} came up.",
        "It felt wrong in {location} after {detail}.",
        "{other} pulled back once {detail} surfaced.",
        "The {object} felt heavier because of {detail}.",
        "That was the first crack in their plan.",
        "The room went quiet when {other} mentioned {detail}.",
        "It hit harder than {name} expected, mostly because of {detail}.",
    ]


def _medium_templates() -> list[str]:
    return [
        "{name} found {object} near {location}, and it shifted the day.",
        "{name} stepped into {location} and felt the mood change.",
        "{name} trusted {other}, even though {other} looked unsure.",
        "{name} kept {object} close, like it might explain everything.",
        "{name} heard the rumor again at {location}, and it sounded different.",
        "{name} watched {other} hesitate, then made a choice.",
        "{name} noticed how {location} was emptier than usual.",
        "{name} promised to fix it, not because it was easy, but because it mattered.",
        "{name} told {other} the truth, and it landed like a weight.",
        "{name} admitted {detail}, and {other} didn't argue.",
    ]


def _long_templates() -> list[str]:
    return [
        "When {name} finally met {other} at {location}, the whole story shifted, because {object} was not a clue, it was a warning.",
        "{name} followed the trail back through {location}, and the people there filled in the missing hours one by one.",
        "{other} admitted the plan had failed, and {name} realized the mistake had been theirs from the start.",
        "By the time {name} opened {object}, {other} had already disappeared, leaving only a choice and a mess.",
        "{name} remembered the first time they walked into {location}, and how the promise they made back then now felt dangerous.",
        "The mistake was not just the decision, it was the silence after it, and {name} could feel the cost growing.",
        "{name} kept the secret too long, and when {other} found out at {location}, nothing about their friendship was the same.",
        "The trouble started with {detail}, and {name} could feel the fallout spreading.",
    ]


class StoryState:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.used_names: set[str] = set()
        self.used_locations: set[str] = set()
        self.used_objects: set[str] = set()
        self.characters: list[dict] = []
        self.locations: list[str] = []
        self.objects: list[str] = []
        self.beat_index = 0
        self.phase = "setup"
        self.last_template_key: str | None = None

    def add_character(self, role: str) -> dict:
        name = _pick_unique(_name_pool(), self.used_names)
        trait = _pick_unique(_traits_pool(), set())
        character = {"name": name, "role": role, "trait": trait}
        self.characters.append(character)
        return character

    def add_location(self) -> str:
        location = _pick_unique(_location_pool(), self.used_locations)
        self.locations.append(location)
        return location

    def add_object(self) -> str:
        obj = _pick_unique(_object_pool(), self.used_objects)
        self.objects.append(obj)
        return obj


def _init_story(topic: str) -> StoryState:
    state = StoryState(topic)
    state.add_character("protagonist")
    state.add_character("friend")
    state.add_location()
    state.add_object()
    return state


def _advance_phase(state: StoryState) -> None:
    phase_order = ["setup", "tension", "reveal", "conclusion"]
    current_index = phase_order.index(state.phase)
    if current_index < len(phase_order) - 1:
        state.phase = phase_order[current_index + 1]


def _expand_world(state: StoryState) -> None:
    if len(state.characters) < 6 and random.random() < 0.6:
        state.add_character("new")
    if len(state.locations) < 8 and random.random() < 0.7:
        state.add_location()
    if len(state.objects) < 8 and random.random() < 0.5:
        state.add_object()


def _build_beat(state: StoryState) -> dict:
    beat_type = random.choice(_beat_types())
    if state.phase == "setup":
        beat_type = random.choice(["setup", "inciting"])
    elif state.phase == "tension":
        beat_type = random.choice(["complication", "pressure", "turn"])
    elif state.phase == "reveal":
        beat_type = random.choice(["reveal", "consequence", "turn"])
    else:
        beat_type = random.choice(["resolution", "aftermath"]) 

    if random.random() < 0.25:
        _expand_world(state)

    character = random.choice(state.characters)
    other = random.choice([c for c in state.characters if c != character])
    location = random.choice(state.locations) if state.locations else state.add_location()
    obj = random.choice(state.objects) if state.objects else state.add_object()
    beat = {
        "type": beat_type,
        "name": character["name"],
        "other": other["name"],
        "location": location,
        "object": obj,
        "detail": random.choice(
            [
                "a promise that felt heavier than it sounded",
                "a plan that slipped out of control",
                "a secret nobody wanted to carry",
                "a sudden risk that had no easy exit",
                "a quiet warning that went ignored",
                "a rumor with too much truth in it",
                "a fragile truce that could snap",
            ]
        ),
    }
    state.beat_index += 1
    if state.beat_index % 12 == 0:
        _advance_phase(state)
    return beat


def _sentence_from_beat(state: StoryState, beat: dict, used_sentences: list[str]) -> str:
    attempts = 0
    while attempts < 50:
        length_choice = random.random()
        if length_choice < 0.2:
            template = random.choice(_short_sentences())
            sentence = template.format(
                name=beat["name"],
                other=beat["other"],
                location=beat["location"],
                object=beat["object"],
                detail=beat["detail"],
            )
            key = "short"
        elif length_choice < 0.65:
            template = random.choice(_medium_templates())
            sentence = template.format(
                name=beat["name"],
                other=beat["other"],
                location=beat["location"],
                object=beat["object"],
                detail=beat["detail"],
            )
            key = "medium"
        else:
            template = random.choice(_long_templates())
            sentence = template.format(
                name=beat["name"],
                other=beat["other"],
                location=beat["location"],
                object=beat["object"],
                detail=beat["detail"],
            )
            key = "long"

        attempts += 1
        if state.last_template_key == key:
            continue
        if _is_repetitive(sentence, used_sentences):
            continue
        state.last_template_key = key
        return sentence

    for _ in range(4):
        _expand_world(state)
        fallback = (
            f"{beat['name']} introduced {state.add_character('ally')['name']} at {state.add_location()}, "
            f"and the story took a new turn with {state.add_object()} and {beat['detail']}."
        )
        if not _is_repetitive(fallback, used_sentences):
            state.last_template_key = "fallback"
            return fallback

    state.last_template_key = "fallback"
    return fallback


def _story_seed(topic: str) -> list[str]:
    return [
        f"There is a story about {topic}, but it starts with a small moment.",
        "The kind you almost skip past.",
    ]


def generate_script(min_seconds: int = MIN_AUDIO_SECONDS) -> ScriptResult:
    topic = random.choice(
        [
            "a city project that went quiet overnight",
            "a neighborhood app that suddenly turned sour",
            "a public promise that kept slipping",
            "a tiny startup that made a giant mistake",
            "a group of friends caught in a slow, messy change",
        ]
    )
    state = _init_story(topic)
    used_sentences: list[str] = []
    lines: list[str] = []

    for seed in _story_seed(topic):
        lines.append(seed)
        used_sentences.append(seed)

    while _estimate_seconds(" ".join(lines)) < float(min_seconds):
        beat = _build_beat(state)
        sentence = _sentence_from_beat(state, beat, used_sentences)
        used_sentences.append(sentence)
        lines.append(sentence)

        if random.random() < 0.25:
            follow_up = _sentence_from_beat(state, beat, used_sentences)
            used_sentences.append(follow_up)
            lines.append(follow_up)

        if random.random() < 0.1:
            lines.append("")

    script = "\n".join(lines).strip()
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
