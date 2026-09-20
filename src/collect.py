"""Стадия collect: источник → data/raw.jsonl.

Источник — HuggingFace датасет pavelfedortsov/russian-colloquial-sft-50k.
Переработка:
  1. генерируем id из хэша user-сообщения;
  2. генерируем topic эвристикой по ключевым словам;
  3. вставляем system-промпт из collect.system_prompts по id.
"""

import hashlib
import json
import time
from pathlib import Path

from datasets import load_dataset

from src.config import load_params


TOPIC_KEYWORDS = {
    "politics_kz":  ["казахстан", "кыргыз", "қазақ", "назарбаев", "жапаров"],
    "politics_ru":  ["росси", "путин", "кремл", "совок", "ссср", "хрущев"],
    "politics_world": ["украин", "израил", "палестин", "хамас", "албани"],
    "tech":         ["телефон", "аккаунт", "телеграм", "приложен", "сайт", "программ", "нейросет", "код", "пароль"],
    "religion":     ["аллах", "бог", "церков", "мечет", "религ", "молитв", "халяль"],
    "math":         ["аргумент", "корень", "формул", "математик", "функци", "уравнен", "геометри"],
    "daily":        ["квартир", "работ", "учёб", "универ", "семь", "друг", "родственник", "аренд", "зарплат"],
    "culture":      ["фильм", "книг", "стих", "музык", "песн", "литератур", "борат"],
    "emotion":      ["любл", "нравит", "свет", "звезд", "красот", "сердц", "чувств"],
}


def guess_topic(text: str) -> str:
    low = text.lower()
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(kw in low for kw in kws):
            return topic
    return "other"


def make_id(user_text: str, idx: int) -> str:
    digest = hashlib.sha1(user_text.encode("utf-8")).hexdigest()[:12]
    return f"colq_{idx:06d}_{digest}"


def pick_prompt(example_id: str, variants: list[str]) -> str:
    digest = hashlib.sha1(example_id.encode("utf-8")).hexdigest()
    return variants[int(digest, 16) % len(variants)]


def main() -> None:
    params = load_params()
    cfg = params["collect"]
    paths = params["paths"]
    n_rows = cfg["n_rows"]
    variants = cfg["system_prompts"]
    if not variants:
        raise SystemExit("collect.system_prompts пуст: инструкцию брать неоткуда")

    out = Path(paths["raw"])
    out.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    scanned = written = 0
    prompts_used: set[str] = set()
    topics_used: dict[str, int] = {}

    ds = load_dataset(cfg["sources"], split="train", streaming=True)

    with out.open("w", encoding="utf-8") as fh:
        for row in ds:
            if written >= n_rows:
                break
            scanned += 1
            msgs = row["messages"]
            if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
                continue

            user_text = msgs[0]["content"]
            assistant_text = msgs[1]["content"]

            example_id = make_id(user_text, written)
            topic = guess_topic(user_text)
            prompt = pick_prompt(example_id, variants)
            prompts_used.add(prompt)
            topics_used[topic] = topics_used.get(topic, 0) + 1

            record = {
                "id": example_id,
                "topic": topic,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    metrics = {
        "version": cfg["version"],
        "sources": cfg["sources"],
        "rows_scanned": scanned,
        "rows_written": written,
        "system_prompt_variants": len(prompts_used),
        "topics": topics_used,
        "seconds": round(time.perf_counter() - started, 2),
    }
    mpath = Path(paths["metrics_collect"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"collect: {written} строк из {scanned} просмотренных, "
        f"вариантов инструкции {len(prompts_used)}, тем {len(topics_used)}, "
        f"{metrics['seconds']} с → {out}"
    )


if __name__ == "__main__":
    main()