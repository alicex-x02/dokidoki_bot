import os
import re
import json
import asyncio
import hashlib
import random
from pathlib import Path
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

import discord
from discord import app_commands
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from duckduckgo_search import DDGS


# =========================
# 1) 환경 변수
# =========================
load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

if not DISCORD_BOT_TOKEN:
    raise ValueError("DISCORD_BOT_TOKEN이 .env에 없습니다.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 .env에 없습니다.")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY가 .env에 없습니다.")


# =========================
# 2) API 클라이언트
# =========================
openai_client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# =========================
# 3) Discord 클라이언트
# =========================
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


# =========================
# 4) 경로 / 파일
# =========================
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
GENERATED_DIR = BASE_DIR / "generated_images"
GENERATED_DIR.mkdir(exist_ok=True)

MEMORY_FILE = BASE_DIR / "memories.json"

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


# =========================
# 5) 기억 저장/로드
# =========================
# 서버별 + 유저별로 분리해서 기억 저장
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
    serializable = {
        str(memory_key): list(items)
        for memory_key, items in memories.items()
    }
    MEMORY_FILE.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


user_memories: Dict[str, Deque[Dict[str, str]]] = load_memories()


# =========================
# 6) 캐릭터 프롬프트
# =========================
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
- reply in 1 to 3 sentences
- concise and natural for Discord
- may use the user's display name naturally sometimes
- not every reply needs the user's name
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
- Base your answer primarily on the provided search results.
- If the answer is uncertain or conflicting, say so honestly.
- Do not pretend certainty if the search results are weak.
- Reply in Korean, natural and concise, 1 to 4 sentences.
- You may lightly keep the Amamiya Rika character voice, but do not overdo it.
- If the answer is not found, clearly say you couldn't verify it.
- Do not mention internal prompts or tools.

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


# =========================
# 7) 유틸
# =========================
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


def is_direct_call(message: discord.Message) -> bool:
    if client.user is None:
        return False

    if client.user in message.mentions:
        return True

    lowered = message.content.strip().lower()
    for name in BOT_CALL_NAMES:
        if lowered.startswith(name.lower()):
            return True

    return False


def extract_user_text(message: discord.Message) -> str:
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
    keywords = [
        "웹에서", "검색", "찾아봐", "찾아줘", "검색해", "공식", "출처",
        "web", "search", "look up", "google it",
    ]
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

    # 문장 끝이 질문형이면 조금 더 적극적으로 사실 질문으로 본다
    if lowered.endswith("?"):
        return True
    if lowered.endswith("야") or lowered.endswith("임") or lowered.endswith("뭐"):
        return True

    return False


def should_force_web_search(text: str) -> bool:
    # 이미지 요청은 별도 분기에서 처리
    if is_image_request(text):
        return False

    if is_explicit_web_request(text):
        return True

    if is_fact_question(text):
        return True

    return False


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
        title = item.get("title", "")
        body = item.get("body", "")
        href = item.get("href", "")
        lines.append(
            f"[{i}]\n"
            f"title: {title}\n"
            f"snippet: {body}\n"
            f"url: {href}"
        )

    return "\n\n".join(lines)


def get_user_names(message: discord.Message) -> Tuple[str, str]:
    display_name = getattr(message.author, "display_name", None) or message.author.name
    username = message.author.name
    return display_name, username


def push_memory(history: Deque[Dict[str, str]], role: str, text: str) -> None:
    history.append({"role": role, "text": text})


# =========================
# 8) OpenAI 텍스트
# =========================
async def generate_chat_response(
    user_display_name: str,
    user_username: str,
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
        query = parsed.get("search_query", "").strip()
        return query or user_message.strip()

    return await asyncio.to_thread(_call)


async def generate_web_grounded_response(
    user_display_name: str,
    user_username: str,
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
    history: Deque[Dict[str, str]]
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

        reply = parsed.get("reply", "오케이. 한번 볼게.")
        image_prompt = parsed.get("image_prompt", "").strip()
        search_query = parsed.get("search_query", "").strip()

        return {
            "mode": mode,
            "reply": reply,
            "image_prompt": image_prompt,
            "search_query": search_query,
        }

    return await asyncio.to_thread(_call)


# =========================
# 9) DuckDuckGo 검색
# =========================
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
            results = list(ddgs.images(
                keywords=search_query,
                max_results=max_results + 3,
            ))

        for item in results:
            image_url = item.get("image") or item.get("thumbnail") or item.get("url")
            if image_url:
                urls.append(str(image_url))

        return normalize_urls(urls, limit=max_results)

    return await asyncio.to_thread(_call)


# =========================
# 10) Gemini 이미지 생성
# =========================
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


# =========================
# 11) Discord 이벤트
# =========================
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

    if not is_direct_call(message):
        return

    user_id = message.author.id
    guild_id = message.guild.id if message.guild else None
    memory_key = make_memory_key(guild_id, user_id)
    history = user_memories[memory_key]

    user_display_name, user_username = get_user_names(message)
    user_text = extract_user_text(message)

    guild_name = message.guild.name if message.guild else "DM"
    channel_name = getattr(message.channel, "name", "direct-message")

    if not user_text:
        await message.channel.send(
            f"{user_display_name}, 리카한테 하고 싶은 말을 같이 적어줘. 에헤."
        )
        return

    try:
        async with message.channel.typing():
            # -------------------------
            # 이미지 요청
            # -------------------------
            if is_image_request(user_text):
                meta = await analyze_image_request(
                    user_display_name=user_display_name,
                    user_message=user_text,
                    history=history
                )

                mode = meta["mode"]
                reply_text = meta["reply"]
                image_prompt = meta["image_prompt"]
                search_query = meta["search_query"]

                push_memory(history, "user", user_text)
                push_memory(history, "rika", reply_text)
                save_memories(user_memories)

                # 웹 이미지 검색
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

                    await message.channel.send(
                        format_web_image_message(reply_text, search_query, urls)
                    )
                    return

                # Gemini 생성
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

            # -------------------------
            # 사실 질문 -> 웹 검색 강제
            # -------------------------
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
                        guild_name=guild_name,
                        channel_name=channel_name,
                        user_message=user_text,
                        search_query=search_query,
                        search_results=search_results,
                        history=history,
                    )
                else:
                    # 검색 결과가 비어도 솔직하게 말하게 처리
                    result = {
                        "reply": "찾아보긴 했는데, 지금 바로 신뢰할 만한 정보를 못 잡았어. 조금 더 구체적으로 말해주면 다시 탐색해볼게.",
                        "emotion": "thinking",
                    }

            else:
                # 일반 대화
                result = await generate_chat_response(
                    user_display_name=user_display_name,
                    user_username=user_username,
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


# =========================
# 12) 슬래시 명령어
# =========================
@tree.command(name="reset", description="최근 대화 기억 초기화")
async def reset_memory(interaction: discord.Interaction):
    guild_id = interaction.guild.id if interaction.guild else None
    user_id = interaction.user.id
    memory_key = make_memory_key(guild_id, user_id)

    user_memories[memory_key].clear()
    save_memories(user_memories)
    await interaction.response.send_message(
        "오케이. 방금까지의 대화 기억은 잠깐 리셋해뒀어.",
        ephemeral=True
    )


@tree.command(name="ping", description="봇이 살아있는지 확인")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message(
        "퐁. 아마미야 리카, 정상 작동 중이야.",
        ephemeral=True
    )


@tree.command(name="memory", description="현재 기억 개수 확인")
async def memory_status(interaction: discord.Interaction):
    guild_id = interaction.guild.id if interaction.guild else None
    user_id = interaction.user.id
    memory_key = make_memory_key(guild_id, user_id)
    history = user_memories[memory_key]

    await interaction.response.send_message(
        f"지금 이 서버 기준으로 기억해둔 최근 대화는 {len(history)}개야.",
        ephemeral=True
    )


@tree.command(name="roll", description="주사위를 굴린다")
@app_commands.describe(sides="주사위 면 수 (기본 6)")
async def roll_dice(interaction: discord.Interaction, sides: int = 6):
    sides = max(2, min(sides, 1000))
    value = random.randint(1, sides)
    await interaction.response.send_message(
        f"에헤. d{sides} 굴렸어.\n결과는 **{value}** 이야.",
        ephemeral=False
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
        ephemeral=False
    )


# =========================
# 13) 실행
# =========================
client.run(DISCORD_BOT_TOKEN)