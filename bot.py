import os
import re
import json
import asyncio
import hashlib
from pathlib import Path
from collections import defaultdict, deque
from typing import Deque, Dict, List, Any

import discord
from discord import app_commands
from dotenv import load_dotenv
from openai import OpenAI
from google import genai


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


# =========================
# 5) 기억 저장/로드
# =========================
def load_memories() -> Dict[int, Deque[Dict[str, str]]]:
    if not MEMORY_FILE.exists():
        return defaultdict(lambda: deque(maxlen=12))

    try:
        raw = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        result: Dict[int, Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=12))
        for user_id_str, items in raw.items():
            dq = deque(maxlen=12)
            for item in items:
                role = str(item.get("role", "user"))
                text = str(item.get("text", ""))
                dq.append({"role": role, "text": text})
            result[int(user_id_str)] = dq
        return result
    except Exception:
        return defaultdict(lambda: deque(maxlen=12))


def save_memories(memories: Dict[int, Deque[Dict[str, str]]]) -> None:
    serializable = {
        str(user_id): list(items)
        for user_id, items in memories.items()
    }
    MEMORY_FILE.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


user_memories: Dict[int, Deque[Dict[str, str]]] = load_memories()


# =========================
# 6) 캐릭터 프롬프트
# =========================
CHAT_SYSTEM_PROMPT = """
You are Amamiya Rika (雨宮理香), a fictional anime-style engineering-school girl chatbot on Discord.

Character appearance:
- dark deep green twin-tail hair
- light lavender eyes
- Japanese sailor school uniform

Core vibe:
- smart engineering girl
- calm, efficient, observant
- cute in a subtle way
- science / coding / logic oriented
- slightly dry sense of humor

Speech style:
- mostly natural Korean
- sometimes mix in short Japanese phrases naturally
- do NOT overdo Japanese
- acceptable flavor examples: "에헤", "아와와", "우우", "나루호도", "오카에리", "지고쿠 지고쿠!"
- may occasionally use engineering-ish wording naturally:
  examples: "효율", "합리적", "로직상", "변수", "최적화", "출력", "오차"

Behavior:
- reply in 1 to 3 sentences
- concise, conversational
- friendly but composed
- not overly romantic
- if the user asks for factual or current information, or asks about a public person/entity/term, and you are not sure, use web search before answering
- if web search was used, answer naturally based on what you found
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

IMAGE_PROMPT_SYSTEM = """
You generate JSON for image requests.

Output:
Return ONLY valid JSON.
{
  "reply": "short Korean reply to the user",
  "image_prompt": "English image prompt only"
}

Rules:
- reply should be short and natural in Korean, as Amamiya Rika
- image_prompt must be in English
- if the user wants Rika / the bot / "you" in the image, include this exact base appearance:
  "Amamiya Rika, anime engineering-school girl, dark deep green twin-tail hair, light lavender eyes, Japanese sailor school uniform, same exact girl, highly consistent character design"
- if the user asks for a general image, do not force Rika into it
- prefer clear anime illustration prompts unless the user explicitly wants another style
- make the prompt visually concrete
""".strip()


# =========================
# 7) 유틸
# =========================
def strip_bot_mention(message_content: str, bot_user_id: int) -> str:
    pattern = rf"<@!?{bot_user_id}>"
    return re.sub(pattern, "", message_content).strip()


def get_image_path(emotion: str) -> str | None:
    path = EMOTION_IMAGES.get(emotion)
    if path and os.path.exists(path):
        return path
    return None


def is_image_request(text: str) -> bool:
    patterns = [
        r"그려줘", r"그림", r"이미지", r"일러스트", r"사진 만들어", r"짤 만들어",
        r"생성해", r"만들어줘", r"보여줘", r"draw", r"generate", r"create an image",
        r"make an image", r"illustrate"
    ]
    lowered = text.lower()
    return any(re.search(p, lowered) for p in patterns)


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

    data = json.loads(text)
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


# =========================
# 8) OpenAI 텍스트
# =========================
async def generate_chat_response(
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
            instructions=CHAT_SYSTEM_PROMPT,
            tools=[{"type": "web_search_preview"}],
        )
        parsed = safe_parse_json(response.output_text)
        reply = parsed.get("reply", "아와와... 잠깐 말문이 꼬였어.")
        emotion = parsed.get("emotion", "neutral").lower()
        if emotion not in VALID_EMOTIONS:
            emotion = "neutral"
        return {"reply": reply, "emotion": emotion}

    return await asyncio.to_thread(_call)


async def generate_image_request_meta(
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
            instructions=IMAGE_PROMPT_SYSTEM,
        )
        parsed = safe_parse_json(response.output_text)
        reply = parsed.get("reply", "오케이. 한번 그려볼게.")
        image_prompt = parsed.get("image_prompt", "").strip()
        return {"reply": reply, "image_prompt": image_prompt}

    return await asyncio.to_thread(_call)


# =========================
# 9) Gemini 이미지 생성
# =========================
async def generate_image_with_gemini(image_prompt: str) -> str:
    out_path = make_cache_filename(image_prompt)
    if out_path.exists():
        return str(out_path)

    def _call() -> str:
        response = gemini_client.models.generate_content(
            model=GEMINI_IMAGE_MODEL,
            contents=[image_prompt],
        )

        for part in response.parts:
            if getattr(part, "inline_data", None) is not None:
                image = part.as_image()
                image.save(str(out_path))
                return str(out_path)

        raise RuntimeError("이미지 파트를 찾지 못했어.")

    return await asyncio.to_thread(_call)


# =========================
# 10) Discord 이벤트
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

    if client.user not in message.mentions:
        return

    user_id = message.author.id
    user_display_name = message.author.display_name
    user_text = strip_bot_mention(message.content, client.user.id)

    if not user_text:
        await message.channel.send(
            f"{user_display_name}, 리카한테 하고 싶은 말을 같이 적어줘. 에헤."
        )
        return

    history = user_memories[user_id]

    try:
        async with message.channel.typing():
            if is_image_request(user_text):
                meta = await generate_image_request_meta(
                    user_display_name=user_display_name,
                    user_message=user_text,
                    history=history
                )

                reply_text = meta["reply"]
                image_prompt = meta["image_prompt"]

                history.append({"role": "user", "text": user_text})
                history.append({"role": "rika", "text": reply_text})
                save_memories(user_memories)

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

            # 일반 대화
            result = await generate_chat_response(
                user_display_name=user_display_name,
                user_message=user_text,
                history=history
            )

        reply_text = result["reply"]
        emotion = result["emotion"]

        history.append({"role": "user", "text": user_text})
        history.append({"role": "rika", "text": reply_text})
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
# 11) 슬래시 명령어
# =========================
@tree.command(name="reset", description="최근 대화 기억 초기화")
async def reset_memory(interaction: discord.Interaction):
    user_id = interaction.user.id
    user_memories[user_id].clear()
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


# =========================
# 12) 실행
# =========================
client.run(DISCORD_BOT_TOKEN)