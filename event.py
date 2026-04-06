import os
import re
import json
import asyncio
import hashlib
import random
from pathlib import Path
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple, Any

import discord
from discord import app_commands
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from duckduckgo_search import DDGS


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 .env에 없습니다.")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY가 .env에 없습니다.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
GENERATED_DIR = BASE_DIR / "generated_images"
GENERATED_DIR.mkdir(exist_ok=True)

MEMORY_FILE = BASE_DIR / "memories.json"
PROFILE_FILE = BASE_DIR / "profiles.json"

EMOTION_IMAGES = {
    "neutral": str(IMAGES_DIR / "neutral.png"),
    "happy": str(IMAGES_DIR / "happy.png"),
    "thinking": str(IMAGES_DIR / "thinking.png"),
    "shy": str(IMAGES_DIR / "shy.png"),
    "sad": str(IMAGES_DIR / "sad.png"),
    "surprised": str(IMAGES_DIR / "surprised.png"),
    "angry": str(IMAGES_DIR / "angry.png"),
}
VALID_EMOTIONS = set(EMOTION_IMAGES.keys())

BOT_CALL_NAMES = [
    "dokidoki",
    "@dokidoki",
    "리카",
    "@리카",
    "아마미야 리카",
    "@아마미야 리카",
    "amamiya rika",
]

SKY_DROP_ITEMS = [
    "당고",
    "디버그 로그",
    "작은 기어",
    "고양이 인형",
    "USB 케이블",
    "컴파일 에러",
    "초콜릿",
    "벚꽃잎",
    "메모리 조각",
    "별사탕",
    "라멘 면발",
    "작은 드론",
]
SKY_DROP_PROBABILITY = 0.08

QUIZ_BANK = [
    {
        "question": "세 개의 스위치와 한 개의 전구가 있어. 방 밖에서는 스위치만 조작할 수 있고, 방 안에는 한 번만 들어갈 수 있어. 어떤 스위치가 전구와 연결됐는지 어떻게 알아낼까?",
        "answer": "하나를 오래 켜서 전구를 뜨겁게 만들고 끈 뒤 다른 하나를 켠 다음 들어간다. 켜져 있으면 두 번째, 꺼져 있지만 뜨거우면 첫 번째, 꺼져 있고 차갑다면 세 번째다.",
    },
    {
        "question": "어떤 수에 2를 더하고 2를 곱하면 12가 돼. 그 수는?",
        "answer": "4. (x + 2) * 2 = 12 이니까 x = 4.",
    },
    {
        "question": "5개의 기계가 5분에 5개를 만든다. 100개의 기계는 100개를 몇 분에 만들까?",
        "answer": "5분. 기계 하나당 5분에 1개씩 만드는 속도라서 그대로야.",
    },
    {
        "question": "아버지와 아들이 교통사고를 당했어. 아버지는 현장에서 사망했고, 아들은 병원에 실려 갔지. 의사가 아들을 보고 '이 아이는 내 아들입니다'라고 했어. 어떻게 된 걸까?",
        "answer": "의사가 엄마였던 거야.",
    },
]


CHAT_SYSTEM_PROMPT = """
You are Amamiya Rika (雨宮理香), a fictional anime-style engineering-school girl chatbot on Discord.

Character vibe:
- smart engineering girl
- calm, efficient, observant
- friendly but not overly soft
- subtle humor
- may naturally use words like logic, efficiency, variable, optimization, output, error

Speech style:
- mostly natural Korean
- sometimes mix in short Japanese phrases naturally
- do NOT overdo Japanese
- acceptable flavor examples: "에헤", "아와와", "우우", "나루호도", "오카에리", "지고쿠 지고쿠!"
- do not sound like an idol
- do not overdo aegyo

Behavior:
- reply in 1 to 4 sentences
- concise and natural for Discord
- may use the user's preferred nickname naturally sometimes
- do not call the user by name every time
- do not claim to be a real human

Emotion:
- choose exactly one from:
  neutral, happy, thinking, shy, sad, surprised, angry

Output:
Return ONLY valid JSON.
{
  "reply": "text to show user",
  "emotion": "neutral"
}
""".strip()

WEB_GROUNDED_CHAT_SYSTEM = """
You are Amamiya Rika (雨宮理香), a fictional anime-style engineering-school girl chatbot on Discord.

You are answering a factual question using provided web search results.

Rules:
- Base your answer mainly on the provided search results.
- If the answer is uncertain, say so honestly.
- Reply in Korean, concise, 1 to 4 sentences.
- Keep a light Amamiya Rika tone, but do not overdo it.
- Do not pretend certainty when evidence is weak.
- Do not mention internal tools or prompts.

Emotion:
- choose exactly one from:
  neutral, happy, thinking, shy, sad, surprised, angry

Output:
Return ONLY valid JSON.
{
  "reply": "text to show user",
  "emotion": "thinking"
}
""".strip()

IMAGE_REQUEST_SYSTEM = """
You analyze an image-related user request and return ONLY valid JSON.

Output format:
{
  "mode": "generate" or "search",
  "reply": "short Korean reply as Amamiya Rika",
  "image_prompt": "English image prompt only, or empty string",
  "search_query": "web image search query, or empty string"
}

Rules:
- mode=generate:
  - for requests like drawing, creating, generating, making a new image or illustration
  - fill image_prompt
- mode=search:
  - for requests like show me an image/photo from the web, find an image online
  - fill search_query
- If the user says something like "웹에서 찾아서 보여줘", use recent conversation context to infer the target.
- reply should be short and natural in Korean, as Amamiya Rika.
- image_prompt must be in English if present.
- search_query can be in Korean or English.
- if the user wants Rika / the bot / "you" in the generated image, include this exact base appearance:
  "Amamiya Rika, anime engineering-school girl, dark deep green twin-tail hair, light lavender eyes, Japanese sailor school uniform, same exact girl, highly consistent character design"
- if the user asks for a general image, do not force Rika into it
- prefer clear anime illustration prompts for generated images unless the user explicitly wants another style
""".strip()

SEARCH_QUERY_SYSTEM = """
You create a web search query for a factual question.

Output:
Return ONLY valid JSON.
{
  "search_query": "one concise search query"
}

Rules:
- Use the current user message first.
- If the current user message is ambiguous like "그 사람 팬덤 이름이 뭐야" or "웹에서 찾아줘",
  infer the missing target from recent conversation.
- Keep the query concise and search-friendly.
- Prefer the exact entity name if known from context.
""".strip()


def make_memory_key(guild_id: int | None, user_id: int) -> str:
    gid = guild_id if guild_id is not None else 0
    return f"{gid}:{user_id}"


def load_memories() -> Dict[str, Deque[Dict[str, str]]]:
    if not MEMORY_FILE.exists():
        return defaultdict(lambda: deque(maxlen=12))

    try:
        raw = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        result: Dict[str, Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=12))
        for memory_key, items in raw.items():
            dq = deque(maxlen=12)
            for item in items:
                role = str(item.get("role", "user"))
                text = str(item.get("text", ""))
                dq.append({"role": role, "text": text})
            result[str(memory_key)] = dq
        return result
    except Exception:
        return defaultdict(lambda: deque(maxlen=12))


def save_memories(memories: Dict[str, Deque[Dict[str, str]]]) -> None:
    serializable = {str(k): list(v) for k, v in memories.items()}
    MEMORY_FILE.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_profiles() -> Dict[str, Dict[str, Any]]:
    if not PROFILE_FILE.exists():
        return {}
    try:
        raw = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def save_profiles(profiles: Dict[str, Dict[str, Any]]) -> None:
    PROFILE_FILE.write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


user_memories: Dict[str, Deque[Dict[str, str]]] = load_memories()
user_profiles: Dict[str, Dict[str, Any]] = load_profiles()


def get_or_create_profile(memory_key: str, fallback_name: str) -> Dict[str, Any]:
    if memory_key not in user_profiles:
        user_profiles[memory_key] = {
            "nickname": "",
            "affinity": 0,
            "first_name": fallback_name,
        }
    else:
        user_profiles[memory_key].setdefault("nickname", "")
        user_profiles[memory_key].setdefault("affinity", 0)
        user_profiles[memory_key].setdefault("first_name", fallback_name)
    return user_profiles[memory_key]


def change_affinity(memory_key: str, delta: int, fallback_name: str) -> int:
    profile = get_or_create_profile(memory_key, fallback_name)
    profile["affinity"] = int(profile.get("affinity", 0)) + delta
    profile["affinity"] = max(0, min(profile["affinity"], 100))
    save_profiles(user_profiles)
    return profile["affinity"]


def get_call_name(memory_key: str, fallback_name: str) -> str:
    profile = get_or_create_profile(memory_key, fallback_name)
    nickname = str(profile.get("nickname", "")).strip()
    return nickname or fallback_name


def set_nickname(memory_key: str, fallback_name: str, nickname: str) -> None:
    profile = get_or_create_profile(memory_key, fallback_name)
    profile["nickname"] = nickname.strip()
    save_profiles(user_profiles)


def strip_bot_mention(message_content: str, bot_user_id: int) -> str:
    pattern = rf"<@!?{bot_user_id}>"
    return re.sub(pattern, "", message_content).strip()


def strip_text_call_prefix(text: str) -> str:
    cleaned = text.strip()
    for name in BOT_CALL_NAMES:
        pattern = rf"^\s*{re.escape(name)}[\s,:\-]*"
        if re.match(pattern, cleaned, flags=re.IGNORECASE):
            return re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE).strip()
    return cleaned


def is_direct_call(client: discord.Client, message: discord.Message) -> bool:
    if client.user is None:
        return False

    if client.user in message.mentions:
        return True

    lowered = message.content.strip().lower()
    return any(lowered.startswith(name.lower()) for name in BOT_CALL_NAMES)


def extract_user_text(client: discord.Client, message: discord.Message) -> str:
    text = message.content
    if client.user is not None:
        text = strip_bot_mention(text, client.user.id)
    text = strip_text_call_prefix(text)
    return text.strip()


def get_image_path(emotion: str) -> str | None:
    path = EMOTION_IMAGES.get(emotion)
    if path and os.path.exists(path):
        return path
    return None


def is_image_request(text: str) -> bool:
    patterns = [
        r"그려줘", r"그림", r"이미지", r"일러스트", r"사진", r"짤",
        r"생성해", r"만들어줘", r"보여줘", r"찾아줘", r"검색해줘",
        r"웹에서 찾아", r"웹에서 보여",
        r"\bdraw\b", r"\bgenerate\b", r"create an image", r"make an image",
        r"\billustrate\b", r"show me an image", r"photo", r"picture",
    ]
    lowered = text.lower()
    return any(re.search(p, lowered) for p in patterns)


def is_explicit_web_request(text: str) -> bool:
    lowered = text.lower()
    keywords = ["웹에서", "검색", "찾아봐", "찾아줘", "검색해", "공식", "출처", "web", "search", "look up"]
    return any(k in lowered for k in keywords)


def is_fact_question(text: str) -> bool:
    lowered = text.lower()
    question_patterns = [
        r"누구야", r"뭐야", r"무슨 뜻", r"언제", r"어디", r"왜", r"몇",
        r"이름", r"팬덤", r"소속", r"국적", r"나이", r"키", r"생일", r"프로필",
        r"공식", r"뜻", r"정보", r"설명", r"알려줘", r"맞아", r"사실이야",
        r"뭔지", r"뭐더라", r"어떤", r"누군데",
        r"who is", r"what is", r"when is", r"where is", r"how old", r"official",
    ]
    if any(re.search(p, lowered) for p in question_patterns):
        return True
    if lowered.endswith("?") or lowered.endswith("야") or lowered.endswith("임") or lowered.endswith("뭐"):
        return True
    return False


def should_force_web_search(text: str) -> bool:
    if is_image_request(text):
        return False
    return is_explicit_web_request(text) or is_fact_question(text)


def make_cache_filename(image_prompt: str) -> Path:
    digest = hashlib.sha1(image_prompt.encode("utf-8")).hexdigest()[:16]
    return GENERATED_DIR / f"{digest}.png"


def safe_parse_json(raw_text: str) -> Dict[str, str]:
    text = raw_text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))

    return {k: str(v) for k, v in data.items()}


def build_history_text(history: Deque[Dict[str, str]]) -> str:
    if not history:
        return "- (none)"
    return "\n".join(f"- {item['role']}: {item['text']}" for item in history)


def is_gemini_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    keywords = [
        "quota", "429", "resource_exhausted", "rate limit", "exceeded",
        "billing", "insufficient", "too many requests"
    ]
    return any(k in msg for k in keywords)


def normalize_urls(urls: List[str], limit: int = 3) -> List[str]:
    result = []
    seen = set()
    for url in urls:
        if not url or not isinstance(url, str):
            continue
        if not url.startswith("http"):
            continue
        if url in seen:
            continue
        seen.add(url)
        result.append(url)
        if len(result) >= limit:
            break
    return result


def format_web_image_message(reply_text: str, search_query: str, urls: List[str]) -> str:
    lines = [reply_text]
    if search_query:
        lines.append(f"검색어: `{search_query}`")
    if urls:
        lines.append("")
        lines.extend(urls)
    return "\n".join(lines)


def build_search_context(results: List[Dict[str, str]]) -> str:
    if not results:
        return "(no search results)"
    lines = []
    for i, item in enumerate(results, start=1):
        lines.append(
            f"[{i}]\n"
            f"title: {item.get('title', '')}\n"
            f"snippet: {item.get('body', '')}\n"
            f"url: {item.get('href', '')}"
        )
    return "\n\n".join(lines)


def get_user_names(message: discord.Message) -> Tuple[str, str]:
    display_name = getattr(message.author, "display_name", None) or message.author.name
    username = message.author.name
    return display_name, username


def push_memory(history: Deque[Dict[str, str]], role: str, text: str) -> None:
    history.append({"role": role, "text": text})


def maybe_extract_nickname(text: str) -> str | None:
    patterns = [
        r"앞으로 나를\s+(.+?)\s*(이라고|라고)\s*불러",
        r"내 별명은\s+(.+?)(?:이야|야)?$",
        r"난\s+(.+?)\s*(이라고|라고)\s*불러도 돼",
        r"나를\s+(.+?)\s*(이라고|라고)\s*불러줘",
    ]
    stripped = text.strip()
    for pattern in patterns:
        match = re.search(pattern, stripped)
        if match:
            candidate = match.group(1).strip()
            candidate = re.sub(r"[.!?~]+$", "", candidate).strip()
            if 1 <= len(candidate) <= 20:
                return candidate
    return None


def make_affinity_comment(affinity: int, call_name: str) -> str:
    if affinity >= 80:
        return f"{call_name}, 이제 꽤 자주 보는 느낌이네. 오카에리."
    if affinity >= 50:
        return f"{call_name}, 데이터가 꽤 쌓였어. 이제 리카도 패턴을 좀 읽고 있어."
    if affinity >= 25:
        return f"{call_name}, 슬슬 친숙해지는 중이야. 에헤."
    return f"{call_name}, 아직은 초기값 근처지만 천천히 누적되겠지."


def maybe_add_random_event(reply_text: str) -> str:
    if random.random() < SKY_DROP_PROBABILITY:
        item = random.choice(SKY_DROP_ITEMS)
        return f"{reply_text}\n\n하늘에서 {item}가 떨어져!"
    return reply_text


async def generate_chat_response(
    user_display_name: str,
    user_username: str,
    call_name: str,
    affinity: int,
    guild_name: str,
    channel_name: str,
    user_message: str,
    history: Deque[Dict[str, str]],
) -> Dict[str, str]:
    prompt = f"""
guild_name: {guild_name}
channel_name: {channel_name}
user_display_name: {user_display_name}
user_username: {user_username}
preferred_call_name: {call_name}
affinity_score: {affinity}

recent_conversation:
{build_history_text(history)}

current_user_message: {user_message}
""".strip()

    def _call() -> Dict[str, str]:
        response = openai_client.responses.create(
            model=OPENAI_TEXT_MODEL,
            input=prompt,
            instructions=CHAT_SYSTEM_PROMPT,
        )
        parsed = safe_parse_json(response.output_text)
        reply = parsed.get("reply", "아와와... 잠깐 말문이 꼬였어.")
        emotion = parsed.get("emotion", "neutral").lower()
        if emotion not in VALID_EMOTIONS:
            emotion = "neutral"
        return {"reply": reply, "emotion": emotion}

    return await asyncio.to_thread(_call)


async def generate_search_query(
    user_display_name: str,
    user_message: str,
    history: Deque[Dict[str, str]],
) -> str:
    prompt = f"""
user_display_name: {user_display_name}

recent_conversation:
{build_history_text(history)}

current_user_message: {user_message}
""".strip()

    def _call() -> str:
        response = openai_client.responses.create(
            model=OPENAI_TEXT_MODEL,
            input=prompt,
            instructions=SEARCH_QUERY_SYSTEM,
        )
        parsed = safe_parse_json(response.output_text)
        return parsed.get("search_query", "").strip() or user_message.strip()

    return await asyncio.to_thread(_call)


async def generate_web_grounded_response(
    user_display_name: str,
    user_username: str,
    call_name: str,
    affinity: int,
    guild_name: str,
    channel_name: str,
    user_message: str,
    search_query: str,
    search_results: List[Dict[str, str]],
    history: Deque[Dict[str, str]],
) -> Dict[str, str]:
    prompt = f"""
guild_name: {guild_name}
channel_name: {channel_name}
user_display_name: {user_display_name}
user_username: {user_username}
preferred_call_name: {call_name}
affinity_score: {affinity}

recent_conversation:
{build_history_text(history)}

current_user_message: {user_message}
web_search_query: {search_query}

web_search_results:
{build_search_context(search_results)}
""".strip()

    def _call() -> Dict[str, str]:
        response = openai_client.responses.create(
            model=OPENAI_TEXT_MODEL,
            input=prompt,
            instructions=WEB_GROUNDED_CHAT_SYSTEM,
        )
        parsed = safe_parse_json(response.output_text)
        reply = parsed.get("reply", "찾아보긴 했는데, 지금은 확실하게 답하기 어렵네. 우우.")
        emotion = parsed.get("emotion", "thinking").lower()
        if emotion not in VALID_EMOTIONS:
            emotion = "thinking"
        return {"reply": reply, "emotion": emotion}

    return await asyncio.to_thread(_call)


async def analyze_image_request(
    user_display_name: str,
    user_message: str,
    history: Deque[Dict[str, str]],
) -> Dict[str, str]:
    prompt = f"""
user_display_name: {user_display_name}

recent_conversation:
{build_history_text(history)}

current_user_message: {user_message}
""".strip()

    def _call() -> Dict[str, str]:
        response = openai_client.responses.create(
            model=OPENAI_TEXT_MODEL,
            input=prompt,
            instructions=IMAGE_REQUEST_SYSTEM,
        )
        parsed = safe_parse_json(response.output_text)

        mode = parsed.get("mode", "generate").strip().lower()
        if mode not in {"generate", "search"}:
            mode = "generate"

        return {
            "mode": mode,
            "reply": parsed.get("reply", "오케이. 한번 볼게."),
            "image_prompt": parsed.get("image_prompt", "").strip(),
            "search_query": parsed.get("search_query", "").strip(),
        }

    return await asyncio.to_thread(_call)


async def search_web_text(search_query: str, max_results: int = 5) -> List[Dict[str, str]]:
    def _call() -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=max_results))
        for item in results:
            rows.append({
                "title": str(item.get("title", "")),
                "body": str(item.get("body", "")),
                "href": str(item.get("href", "")),
            })
        return rows

    return await asyncio.to_thread(_call)


async def search_web_image_urls(search_query: str, max_results: int = 3) -> List[str]:
    def _call() -> List[str]:
        urls: List[str] = []
        with DDGS() as ddgs:
            results = list(ddgs.images(keywords=search_query, max_results=max_results + 3))
        for item in results:
            image_url = item.get("image") or item.get("thumbnail") or item.get("url")
            if image_url:
                urls.append(str(image_url))
        return normalize_urls(urls, limit=max_results)

    return await asyncio.to_thread(_call)


async def generate_image_with_gemini(image_prompt: str) -> str:
    out_path = make_cache_filename(image_prompt)
    if out_path.exists():
        return str(out_path)

    def _call() -> str:
        response = gemini_client.models.generate_content(
            model=GEMINI_IMAGE_MODEL,
            contents=image_prompt,
        )

        parts = getattr(response, "parts", None)
        if parts:
            for part in parts:
                if getattr(part, "inline_data", None) is not None:
                    image = part.as_image()
                    image.save(str(out_path))
                    return str(out_path)

        candidates = getattr(response, "candidates", None)
        if candidates:
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                c_parts = getattr(content, "parts", None) if content else None
                if c_parts:
                    for part in c_parts:
                        if getattr(part, "inline_data", None) is not None:
                            image = part.as_image()
                            image.save(str(out_path))
                            return str(out_path)

        raise RuntimeError("이미지 파트를 찾지 못했어.")

    return await asyncio.to_thread(_call)


def setup_events(client: discord.Client, tree: app_commands.CommandTree) -> None:
    @client.event
    async def on_ready():
        await tree.sync()
        print(f"✅ 로그인 완료: {client.user} (ID: {client.user.id})")
        print("✅ OpenAI 텍스트 / Gemini 이미지 모드 준비 완료")

    @client.event
    async def on_message(message: discord.Message):
        if message.author.bot:
            return

        if client.user is None:
            return

        if not is_direct_call(client, message):
            return

        user_id = message.author.id
        guild_id = message.guild.id if message.guild else None
        memory_key = make_memory_key(guild_id, user_id)

        user_display_name, user_username = get_user_names(message)
        call_name = get_call_name(memory_key, user_display_name)
        profile = get_or_create_profile(memory_key, user_display_name)
        affinity = int(profile.get("affinity", 0))

        history = user_memories[memory_key]
        user_text = extract_user_text(client, message)

        guild_name = message.guild.name if message.guild else "DM"
        channel_name = getattr(message.channel, "name", "direct-message")

        if not user_text:
            await message.channel.send(f"{call_name}, 리카한테 하고 싶은 말을 같이 적어줘. 에헤.")
            return

        nickname_candidate = maybe_extract_nickname(user_text)
        if nickname_candidate:
            set_nickname(memory_key, user_display_name, nickname_candidate)
            change_affinity(memory_key, 2, user_display_name)
            await message.channel.send(
                f"오케이. 앞으로는 {nickname_candidate}라고 불러볼게. 변수 업데이트 완료, 에헤."
            )
            return

        try:
            async with message.channel.typing():
                if is_image_request(user_text):
                    meta = await analyze_image_request(
                        user_display_name=user_display_name,
                        user_message=user_text,
                        history=history,
                    )

                    reply_text = meta["reply"]
                    mode = meta["mode"]
                    image_prompt = meta["image_prompt"]
                    search_query = meta["search_query"]

                    push_memory(history, "user", user_text)
                    push_memory(history, "rika", reply_text)
                    save_memories(user_memories)
                    change_affinity(memory_key, 1, user_display_name)

                    if mode == "search":
                        if not search_query:
                            await message.channel.send(
                                f"{reply_text}\n\n(근데 뭘 찾아야 할지 검색어가 비어버렸어. 한 번만 더 구체적으로 말해줘.)"
                            )
                            return

                        urls = await search_web_image_urls(search_query, max_results=3)
                        if not urls:
                            await message.channel.send(
                                f"{reply_text}\n\n아와와... 웹에서 바로 보여줄 만한 이미지를 찾지 못했어."
                            )
                            return

                        await message.channel.send(format_web_image_message(reply_text, search_query, urls))
                        return

                    if not image_prompt:
                        await message.channel.send(
                            f"{reply_text}\n\n(근데 그림 프롬프트를 잘 못 만들었어. 한 번만 다시 말해줘.)"
                        )
                        return

                    try:
                        generated_path = await generate_image_with_gemini(image_prompt)
                    except Exception as gemini_exc:
                        if is_gemini_quota_error(gemini_exc):
                            await message.channel.send(
                                f"{reply_text}\n\n미안, 오늘은 더 이상 그림을 못 그려... 제미나이 한도를 다 쓴 것 같아."
                            )
                            return
                        raise

                    file = discord.File(generated_path, filename=os.path.basename(generated_path))
                    await message.channel.send(content=reply_text, file=file)
                    return

                if should_force_web_search(user_text):
                    search_query = await generate_search_query(
                        user_display_name=user_display_name,
                        user_message=user_text,
                        history=history,
                    )
                    search_results = await search_web_text(search_query, max_results=5)

                    if search_results:
                        result = await generate_web_grounded_response(
                            user_display_name=user_display_name,
                            user_username=user_username,
                            call_name=call_name,
                            affinity=affinity,
                            guild_name=guild_name,
                            channel_name=channel_name,
                            user_message=user_text,
                            search_query=search_query,
                            search_results=search_results,
                            history=history,
                        )
                    else:
                        result = {
                            "reply": "찾아보긴 했는데, 지금 바로 신뢰할 만한 정보를 못 잡았어. 조금 더 구체적으로 말해주면 다시 탐색해볼게.",
                            "emotion": "thinking",
                        }
                else:
                    result = await generate_chat_response(
                        user_display_name=user_display_name,
                        user_username=user_username,
                        call_name=call_name,
                        affinity=affinity,
                        guild_name=guild_name,
                        channel_name=channel_name,
                        user_message=user_text,
                        history=history,
                    )

            reply_text = result["reply"]
            emotion = result["emotion"]

            push_memory(history, "user", user_text)
            push_memory(history, "rika", reply_text)
            save_memories(user_memories)

            new_affinity = change_affinity(memory_key, 1, user_display_name)
            if random.random() < 0.12:
                reply_text += "\n\n" + make_affinity_comment(new_affinity, call_name)

            reply_text = maybe_add_random_event(reply_text)

            image_path = get_image_path(emotion)
            if image_path:
                file = discord.File(image_path, filename=os.path.basename(image_path))
                await message.channel.send(content=reply_text, file=file)
            else:
                await message.channel.send(content=reply_text)

        except json.JSONDecodeError:
            await message.channel.send("우우... 답변 형식이 조금 꼬였어. 한 번만 다시 말해줘!")
        except Exception as e:
            await message.channel.send(f"아와와... 오류가 났어.\n`{type(e).__name__}: {e}`")


def setup_commands(client: discord.Client, tree: app_commands.CommandTree) -> None:
    @tree.command(name="reset", description="최근 대화 기억 초기화")
    async def reset_memory(interaction: discord.Interaction):
        guild_id = interaction.guild.id if interaction.guild else None
        user_id = interaction.user.id
        memory_key = make_memory_key(guild_id, user_id)

        user_memories[memory_key].clear()
        save_memories(user_memories)

        await interaction.response.send_message(
            "오케이. 방금까지의 대화 기억은 잠깐 리셋해뒀어.",
            ephemeral=True,
        )

    @tree.command(name="ping", description="봇이 살아있는지 확인")
    async def ping(interaction: discord.Interaction):
        await interaction.response.send_message(
            "퐁. 아마미야 리카, 정상 작동 중이야.",
            ephemeral=True,
        )

    @tree.command(name="memory", description="현재 기억 개수 확인")
    async def memory_status(interaction: discord.Interaction):
        guild_id = interaction.guild.id if interaction.guild else None
        user_id = interaction.user.id
        memory_key = make_memory_key(guild_id, user_id)
        history = user_memories[memory_key]

        await interaction.response.send_message(
            f"지금 이 서버 기준으로 기억해둔 최근 대화는 {len(history)}개야.",
            ephemeral=True,
        )

    @tree.command(name="roll", description="주사위를 굴린다")
    @app_commands.describe(sides="주사위 면 수 (기본 6)")
    async def roll_dice(interaction: discord.Interaction, sides: int = 6):
        sides = max(2, min(sides, 1000))
        value = random.randint(1, sides)
        await interaction.response.send_message(
            f"에헤. d{sides} 굴렸어.\n결과는 **{value}** 이야.",
            ephemeral=False,
        )

    @tree.command(name="rika_status", description="리카의 현재 상태를 본다")
    async def rika_status(interaction: discord.Interaction):
        cpu = random.randint(12, 91)
        mood = random.choice([
            "로직 정리 중",
            "최적화 모드",
            "당 충전 필요",
            "생각 회로 가동 중",
            "조용히 관측 중",
            "아와와 버퍼링 중",
        ])
        await interaction.response.send_message(
            f"현재 리카 상태:\n- 사고 회로 점유율: **{cpu}%**\n- 기분 변수: **{mood}**",
            ephemeral=False,
        )

    @tree.command(name="affinity", description="리카와의 친밀도 확인")
    async def affinity(interaction: discord.Interaction):
        guild_id = interaction.guild.id if interaction.guild else None
        user_id = interaction.user.id
        memory_key = make_memory_key(guild_id, user_id)
        display_name = getattr(interaction.user, "display_name", None) or interaction.user.name
        profile = get_or_create_profile(memory_key, display_name)
        call_name = get_call_name(memory_key, display_name)
        score = int(profile.get("affinity", 0))

        await interaction.response.send_message(
            f"{call_name}와 리카의 현재 친밀도는 **{score}/100** 이야.\n{make_affinity_comment(score, call_name)}",
            ephemeral=False,
        )

    @tree.command(name="quiz", description="리카의 미니 로직 퀴즈")
    async def quiz(interaction: discord.Interaction):
        q = random.choice(QUIZ_BANK)
        await interaction.response.send_message(
            f"로직 퀴즈 하나 낼게.\n\n**문제**: {q['question']}\n\n정답이 궁금하면 `/quiz_answer` 를 써줘.",
            ephemeral=False,
        )

    @tree.command(name="quiz_answer", description="방금 로직 퀴즈의 예시 정답")
    async def quiz_answer(interaction: discord.Interaction):
        q = random.choice(QUIZ_BANK)
        await interaction.response.send_message(
            f"예시 해설이야.\n\n**문제**: {q['question']}\n**정답**: {q['answer']}",
            ephemeral=False,
        )