import random
from collections import Counter

STOPWORDS = {
    "about",
    "after",
    "before",
    "being",
    "because",
    "between",
    "could",
    "should",
    "there",
    "their",
    "these",
    "those",
    "which",
    "your",
    "with",
    "that",
    "this",
    "what",
    "when",
    "where",
    "who",
    "will",
    "just",
    "into",
    "than",
    "then",
    "have",
    "from",
    "they",
    "them",
    "been",
    "does",
    "every",
    "easy",
    "make",
    "more",
    "most",
    "much",
    "very",
    "like",
}

EMOTION_QUERY_MAP = {
    "uneasy": ["person alone room", "dim bedroom night", "quiet hallway"],
    "comfortable": ["couple laughing indoors", "relaxed home evening", "cozy couch"],
    "confused": ["person thinking window", "staring at phone night", "confused face"],
    "anxious": ["pacing room", "city rain night", "hands nervous"],
    "realisation": ["person sitting still", "deep breath window", "moment of clarity"],
    "reflection": ["sunrise window", "calm street morning", "quiet coffee"],
    "regret": ["looking down alone", "sad person room", "night window"],
    "tension": ["tense face", "hands clenched", "dark room"],
}

FALLBACK_QUERIES = [
    "person alone room",
    "hands nervous",
    "city rain night",
    "quiet street evening",
    "phone closeup",
    "walking alone night",
]


def split_script(script: str, max_lines_per_segment: int = 3) -> list[str]:
    lines = [line.strip() for line in script.splitlines() if line.strip()]
    if not lines:
        return [script]
    segments = []
    for i in range(0, len(lines), max_lines_per_segment):
        segments.append(" ".join(lines[i : i + max_lines_per_segment]))
    return segments


def _tokenize(text: str) -> list[str]:
    tokens = ["".join(char for char in word.lower() if char.isalpha()) for word in text.split()]
    return [token for token in tokens if token and token not in STOPWORDS]


def _top_keywords(tokens: list[str], limit: int = 4) -> list[str]:
    if not tokens:
        return []
    counts = Counter(tokens)
    return [word for word, _ in counts.most_common(limit)]


def _infer_emotion(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ["uneasy", "nervous", "tension", "tense"]):
        return "uneasy"
    if any(word in lowered for word in ["comfortable", "calm", "normal", "safe"]):
        return "comfortable"
    if any(word in lowered for word in ["confused", "unclear", "question", "why"]):
        return "confused"
    if any(word in lowered for word in ["anxious", "anxiety", "panic", "eggshells", "afraid"]):
        return "anxious"
    if any(word in lowered for word in ["realized", "realised", "click", "turning point", "moment"]):
        return "realisation"
    if any(word in lowered for word in ["reflection", "learned", "lesson", "looking back"]):
        return "reflection"
    if any(word in lowered for word in ["regret", "wish", "should have"]):
        return "regret"
    return random.choice(["uneasy", "confused", "anxious", "reflection"])


def extract_queries(text: str, min_queries: int = 3, max_queries: int = 5) -> list[str]:
    emotion = _infer_emotion(text)
    queries = list(EMOTION_QUERY_MAP.get(emotion, []))

    tokens = _tokenize(text)
    keywords = _top_keywords(tokens)
    if keywords:
        queries.append("person " + keywords[0])
        if len(keywords) > 1:
            queries.append("hands " + keywords[1])

    while len(queries) < min_queries:
        queries.append(random.choice(FALLBACK_QUERIES))

    random.shuffle(queries)
    return queries[:max_queries]


def build_query_sets(script_segments: list[str]) -> list[list[str]]:
    return [extract_queries(segment) for segment in script_segments]
