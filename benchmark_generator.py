import json
import re
import time
from pathlib import Path
from typing import Any

from RAG.Rag import Rag


DATA_DIR = Path("KnowledgeBase/synthetic_benchmark")
DATA_DIR.mkdir(parents=True, exist_ok=True)
WORLD_BIBLE_PATH = DATA_DIR / "world_bible.json"

TOTAL_QUESTIONS = 200
MAX_JSON_RETRIES = 3

GENERATOR_MODEL_NAME = "huihui_ai/qwen2.5-abliterate:72b"
GENERATOR_MODEL_MODE = "api"

GENERATOR_SYSTEM_PROMPT = """You are a benchmark dataset generator.
Always follow the requested output format exactly.
When JSON is requested, return valid JSON only.
Do not include markdown fences unless explicitly requested.
"""

WORLD_BIBLE_PROMPT = """You are creating the canonical lore bible for a fictional world called Eldara.

Create a strict world canon as valid JSON only. No markdown, no extra text.
Use this exact top-level structure:
{
    "world_name": "Eldara",
    "timeline": [
        {"year": 0, "event": "..."}
    ],
    "factions": [
        {"name": "...", "description": "...", "alliances": ["..."], "rivals": ["..."]}
    ],
    "places": [
        {"name": "...", "type": "continent|city|region|forest|mountain|sea", "description": "..."}
    ],
    "magic_rules": [
        "..."
    ],
    "hard_constraints": [
        "Facts that must never be contradicted in later documents"
    ]
}

Requirements:
- Internally consistent world facts
- At least 20 timeline events with specific years
- At least 12 factions
- At least 20 named places
- At least 12 hard constraints
"""

WORLD_TOPICS = [
    ("founding_history", "The founding of the Valdrek Empire and its three founding emperors"),
    ("continent_geography", "The five continents of Eldara: geography, climate, and natural borders"),
    ("magic_sources", "The four sources of magic: Living Flame, Cold Void, Deep Tide, and Eternal Wind"),
    ("human_factions", "Humans of Eldara: factions, customs, and regional distribution"),
    ("selvari", "The Selvari: a tree-dwelling elven people of the Great Forest of Thuun"),
    ("gorrak", "The Gorrak: stone giants from the Eternal Mountains"),
    ("mereidi", "The Mereidi: an amphibious people of the Silver Coasts"),
    ("economy_trade", "Eldara's economy: the Draken currency and major trade routes"),
    ("religious_pantheon", "The pantheon of Eldara: eight gods and their domains"),
    ("great_fracture_war", "The Great Fracture: the war that split Eldara 300 years ago"),
    ("artifices", "Artifices: magic-powered machines driving Valdrek industry"),
    ("custodian_order", "The Order of Custodians: guardians of magical balance"),
    ("valdrek_politics", "Political structure of the Valdrek Empire: Arc Council and Solar Throne"),
    ("iconic_creatures", "Iconic creatures of Eldara: Ash Dragon, Mist Wolf, and Glimmerfish"),
    ("major_festivals", "The three grand festivals of Eldara: Night of Flames, High Tide, and The Silence"),
    ("southern_free_kingdoms", "The Southern Free Kingdoms: history, alliances, and conflicts with Valdrek"),
    ("bound_sorcerers", "Bound Sorcerers: initiation process, ranks, and powers"),
    ("fracture_zones", "Fracture Zones: unstable and dangerous regions where magic breaks down"),
    ("dawn_prophecy", "The Dawn Prophecy: an ancient text predicting the end of the current age"),
    ("alchemy", "Alchemy in Eldara: schools, legendary ingredients, and practical applications"),
]

LORE_PROMPT = """You are writing worldbuilding lore for a fictional fantasy world called Eldara.
World canon (absolute truth, do not contradict):
{world_bible}

Write a detailed encyclopedic article about: {topic_desc}

Requirements:
- 400 to 600 words
- At least 8 concrete, verifiable facts (names, dates, numbers, places)
- Formal encyclopedic tone
- Fully fictional but internally consistent
- No markdown titles, only plain prose

The concrete facts are required because they will be used to build benchmark questions.
Include precise details such as exact dates, distances, durations, and named entities.
"""

QUESTIONS_PROMPT = """You are an expert benchmark writer for RAG evaluation.

World canon (for consistency checks):
{world_bible}

Given this encyclopedic document from the fictional world of Eldara:

---
{document}
---

Generate exactly {n_questions} multiple-choice questions with 4 options each, using only explicit information from the document.
Use the world canon only to avoid contradictions in names, places, and timeline.
Do not use canon facts as evidence unless they also appear explicitly in the document.

Rules:
- Each question must have exactly one correct answer
- The 3 incorrect options must be plausible
- Mix difficulty levels
- Avoid near-duplicate questions
- Balance correct answers across A, B, C, and D as much as possible

Return valid JSON only, with this exact structure:
{{
  "questions": [
    {{
      "id": "q001",
      "document_id": "{doc_id}",
      "question": "Question text",
      "options": {{
        "A": "Option A",
        "B": "Option B",
        "C": "Option C",
        "D": "Option D"
      }},
      "correct_answer": "A",
      "explanation": "Brief rationale"
    }}
  ]
}}
"""


def _build_generator_rag() -> Rag:
    rag = Rag(
        retriever_model_name="MockRetriever",
        retriever_model_mode="mock",
        generator_model_name=GENERATOR_MODEL_NAME,
        generator_model_mode=GENERATOR_MODEL_MODE,
        rebuild_index=False,
        system_prompt=GENERATOR_SYSTEM_PROMPT,
    )

    # We explicitly start components to avoid index work and keep generation lightweight.
    rag.start_retriever()
    rag.start_generator()
    return rag


def _strip_markdown_fences(text: str) -> str:
    cleaned = re.sub(r"^```json\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    return cleaned.strip()


def _ask_with_rag(rag: Rag, prompt: str) -> str:
    return rag.prompt(query=prompt, benchmark=False).strip()


def _ask_json_with_retries(rag: Rag, prompt: str, payload_name: str) -> dict[str, Any]:
    last_raw = ""
    for attempt in range(1, MAX_JSON_RETRIES + 1):
        raw = _ask_with_rag(rag, prompt)
        raw = _strip_markdown_fences(raw)
        last_raw = raw

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  Warning: invalid JSON for {payload_name} (attempt {attempt}/{MAX_JSON_RETRIES})")
            continue

        if not isinstance(data, dict):
            print(f"  Warning: invalid payload type for {payload_name} (attempt {attempt}/{MAX_JSON_RETRIES})")
            continue

        return data

    preview = last_raw[:320].replace("\n", " ")
    raise ValueError(f"Could not parse valid JSON for {payload_name}. Last output preview: {preview}")


def _world_bible_to_text(world_bible: dict[str, Any]) -> str:
    return json.dumps(world_bible, ensure_ascii=False, indent=2)


def generate_world_bible(rag: Rag) -> dict[str, Any]:
    print("  Generating world_bible.json...")
    data = _ask_json_with_retries(rag, WORLD_BIBLE_PROMPT, "world_bible")
    return data


def load_or_generate_world_bible(rag: Rag) -> dict[str, Any]:
    if WORLD_BIBLE_PATH.exists():
        print("  Existing world_bible.json found, loading...")
        with open(WORLD_BIBLE_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data

    world_bible = generate_world_bible(rag)
    with open(WORLD_BIBLE_PATH, "w", encoding="utf-8") as handle:
        json.dump(world_bible, handle, ensure_ascii=False, indent=2)
    return world_bible


def generate_document(
    rag: Rag,
    topic_id: str,
    topic_desc: str,
    world_bible_text: str,
) -> dict[str, Any]:
    print(f"  Generating document: {topic_id}...")
    response_text = _ask_with_rag(
        rag,
        LORE_PROMPT.format(topic_desc=topic_desc, world_bible=world_bible_text),
    )

    return {
        "id": topic_id,
        "title": topic_desc,
        "content": response_text,
    }


def generate_questions_for_doc(
    rag: Rag,
    doc: dict[str, Any],
    n_questions: int,
    id_offset: int,
    world_bible_text: str,
) -> list[dict[str, Any]]:
    print(f"  Generating {n_questions} questions for: {doc['id']}...")
    prompt = QUESTIONS_PROMPT.format(
        document=doc["content"],
        n_questions=n_questions,
        doc_id=doc["id"],
        world_bible=world_bible_text,
    )

    data = _ask_json_with_retries(rag, prompt, f"questions_{doc['id']}")

    questions = data.get("questions", [])
    if not isinstance(questions, list):
        raise ValueError(f"Invalid questions payload for {doc['id']}")

    for i, question in enumerate(questions):
        if not isinstance(question, dict):
            continue
        question["id"] = f"q{id_offset + i + 1:03d}"

    return questions


def main():
    print("=" * 60)
    print("GENERATING FICTIONAL BENCHMARK: ELDARA")
    print("=" * 60)

    rag = _build_generator_rag()

    print("\n[1/4] Building world canon...")
    world_bible = load_or_generate_world_bible(rag)
    world_bible_text = _world_bible_to_text(world_bible)
    print("  OK: world_bible.json ready")

    print("\n[2/4] Generating lore documents...")
    documents: list[dict[str, Any]] = []

    for topic_id, topic_desc in WORLD_TOPICS:
        doc_path = DATA_DIR / f"doc_{topic_id}.json"

        if doc_path.exists():
            print(f"  Existing document found: {topic_id}, loading...")
            with open(doc_path, "r", encoding="utf-8") as handle:
                doc = json.load(handle)
        else:
            doc = generate_document(rag, topic_id, topic_desc, world_bible_text)
            with open(doc_path, "w", encoding="utf-8") as handle:
                json.dump(doc, handle, ensure_ascii=False, indent=2)
            time.sleep(0.4)

        documents.append(doc)

    with open(DATA_DIR / "documents.json", "w", encoding="utf-8") as handle:
        json.dump(documents, handle, ensure_ascii=False, indent=2)
    print(f"  OK: {len(documents)} documents stored")

    print(f"\n[3/4] Generating {TOTAL_QUESTIONS} benchmark questions...")
    questions_per_doc = TOTAL_QUESTIONS // len(WORLD_TOPICS)
    remainder = TOTAL_QUESTIONS % len(WORLD_TOPICS)

    all_questions: list[dict[str, Any]] = []
    id_offset = 0

    for idx, doc in enumerate(documents):
        per_doc_path = DATA_DIR / f"questions_{doc['id']}.json"
        amount = questions_per_doc + (1 if idx < remainder else 0)

        if per_doc_path.exists():
            print(f"  Existing questions found for: {doc['id']}, loading...")
            with open(per_doc_path, "r", encoding="utf-8") as handle:
                questions = json.load(handle)
        else:
            questions = generate_questions_for_doc(
                rag,
                doc,
                amount,
                id_offset,
                world_bible_text,
            )
            with open(per_doc_path, "w", encoding="utf-8") as handle:
                json.dump(questions, handle, ensure_ascii=False, indent=2)
            time.sleep(0.4)

        all_questions.extend(questions)
        id_offset += len(questions)

    for idx, question in enumerate(all_questions):
        if not isinstance(question, dict):
            continue
        question["id"] = f"q{idx + 1:03d}"

    with open(DATA_DIR / "questions.json", "w", encoding="utf-8") as handle:
        json.dump(all_questions, handle, ensure_ascii=False, indent=2)
    print(f"  OK: {len(all_questions)} questions stored")

    print("\n[4/4] Dataset stats")
    distribution = {"A": 0, "B": 0, "C": 0, "D": 0}
    for question in all_questions:
        if not isinstance(question, dict):
            continue
        answer = question.get("correct_answer")
        if answer in distribution:
            distribution[answer] += 1

    print(f"  Correct-answer distribution: {distribution}")
    print(f"\nDONE: dataset saved under {DATA_DIR}")
    print(f"  - {WORLD_BIBLE_PATH} (global canon)")
    print(f"  - {DATA_DIR / 'documents.json'} ({len(documents)} docs)")
    print(f"  - {DATA_DIR / 'questions.json'} ({len(all_questions)} questions)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: benchmark generation failed: {exc}")
        raise