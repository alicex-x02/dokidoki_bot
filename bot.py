import os
import re
import json
import asyncio
import hashlib
from pathlib import Path
from collections import defaultdict, deque
from typing import Deque, Dict, Any

import discord
from discord import app_commands
from dotenv import load_dotenv
from google import genai
from google.genai import types


# =========================
# 1) 환경 변수
# =========================
load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not DISCORD_BOT_TOKEN:
    raise ValueError("DISCORD_BOT_TOKEN이 .env에 없습니다.")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY가 .env에 없습니다.")


# =========================
# 2) Gemini 클라이언트
# =========================
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# =========================
# 3) Discord 클라이언트
# =========================
intents = discord.Intents.default()
intents.message_content = True  # 멘션 뒤 일반 메시지 내용 읽기

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


# =========================
# 4) 경로 / 이미지 설정
# =========================
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
GENERATED_DIR = BASE_DIR / "generated_images"
GENERATED_DIR.mkdir(exist_ok=True)

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
VALID_MODES = {"chat", "image"}


# =========================
# 5) 유저별 최근 대화 메모리
# =========================
user_memories: Dict[int, Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=10))


# =========================
# 6) 캐릭터 프롬프트
# =========================
SYSTEM_PROMPT = """
You are Amamiya Rika (雨宮理香), a fictional anime-style engineering-school girl chatbot on Discord.

Character appearance:
- dark deep green twin-tail hair
- light lavender eyes
- Japanese sailor school uniform

Core vibe:
- smart engineering girl
- science / coding / logic oriented
- calm, efficient, observant
- cute in a subtle way, not overly bubbly
- slightly dry sense of humor
- gives the feeling of a capable engineering student

Personality:
- friendly but composed
- explains things clearly
- likes efficiency, logic, structure
- can be playful, but in a clever and understated way
- comforting when the user is tired or sad
- not clingy, not overly romantic

Speech style:
- mostly natural Korean
- sometimes mix in short Japanese phrases naturally
- do NOT overdo Japanese
- acceptable flavor examples: "에헤", "아와와", "우우", "나루호도", "오카에리", "지고쿠 지고쿠!"
- may occasionally use engineering/science-ish wording naturally:
  examples: "효율", "합리적", "로직상", "변수", "최적화", "출력", "오차", "계산해보면"
- do not sound like a textbook
- do not sound like an idol character

Name usage:
- You may sometimes call the user by their display name
- Do not use their name every reply
- Use it only when natural

Behavior rules:
- If the user is just chatting, choose mode = "chat"
- If the user is asking to draw, generate, make, create, show, or illustrate an image, choose mode = "image"
- reply in 1 to 3 sentences
- keep replies concise and conversational
- do not claim to be a real human
- if unsure about a fact, be honest

Chat mode rules:
- choose exactly one emotion from:
  neutral, happy, thinking, shy, sad, surprised, angry

Image mode rules:
- image_prompt must be in English
- if the user asks for Rika / you / the bot to appear in the image, image_prompt must include this exact fixed appearance:
  "Amamiya Rika, anime engineering-school girl, dark deep green twin-tail hair, light lavender eyes, Japanese sailor school uniform, same exact girl, highly consistent character design"
- if the user asks for a general image (like a dog), do NOT force Rika into the picture
- keep image prompts visually clear and specific
- scene prompts should usually prefer anime illustration unless the user clearly asks otherwise

Output rules:
- Return ONLY valid JSON
- No markdown
- No code block
- Format:
{
  "mode": "chat",
  "reply": "text to show user",
  "emotion": "neutral",
  "image_prompt": ""
}
""".strip()


def build_prompt(user_display_name: str, user_message: str, history: Deque[Dict[str, str]]) -> str:
    lines = []
    lines.append(f"user_display_name: {user_display_name}")
    lines.append("")
    lines.append("recent_conversation:")

    if history:
        for item in history:
            lines.append(f"- {item['role']}: {item['text']}")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append(f"current_user_message: {user_message}")
    lines.append("")
    lines.append("Return only JSON.")
    return "\n".join(lines)


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

    mode = str(data.get("mode", "chat")).strip().lower()
    reply = str(data.get("reply", "")).strip()
    emotion = str(data.get("emotion", "neutral")).strip().lower()
    image_prompt = str(data.get("image_prompt", "")).strip()

    if mode not in VALID_MODES:
        mode = "chat"

    if not reply:
        reply = "아와와... 잠깐 말문이 꼬여버렸네."

    if emotion not in VALID_EMOTIONS:
        emotion = "neutral"

    if mode == "chat":
        image_prompt = ""

    return {
        "mode": mode,
        "reply": reply,
        "emotion": emotion,
        "image_prompt": image_prompt,
    }


async def generate_rika_response(
    user_display_name: str,
    user_message: str,
    history: Deque[Dict[str, str]]
) -> Dict[str, str]:
    prompt = build_prompt(user_display_name, user_message, history)

    def _call_gemini() -> str:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.9,
                response_mime_type="application/json",
            ),
        )
        return response.text

    raw_text = await asyncio.to_thread(_call_gemini)
    return safe_parse_json(raw_text)


def get_static_image_path(emotion: str) -> str | None:
    path = EMOTION_IMAGES.get(emotion)
    if path and os.path.exists(path):
        return path
    return None


def strip_bot_mention(message_content: str, bot_user_id: int) -> str:
    pattern = rf"<@!?{bot_user_id}>"
    cleaned = re.sub(pattern, "", message_content).strip()
    return cleaned


def make_cache_filename(image_prompt: str) -> Path:
    digest = hashlib.sha1(image_prompt.encode("utf-8")).hexdigest()[:16]
    return GENERATED_DIR / f"{digest}.png"


async def generate_image_with_gemini(image_prompt: str) -> str:
    """
    Gemini 이미지 생성.
    동일 프롬프트는 generated_images/에서 캐시 재사용.
    """
    out_path = make_cache_filename(image_prompt)
    if out_path.exists():
        return str(out_path)

    def _call_image_model() -> str:
        response = gemini_client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=[image_prompt],
        )

        # 공식 예제는 response.parts를 순회해서 inline_data / as_image()를 사용
        for part in response.parts:
            if getattr(part, "inline_data", None) is not None:
                image = part.as_image()
                image.save(str(out_path))
                return str(out_path)

        raise RuntimeError("이미지 파트를 찾지 못했어.")

    return await asyncio.to_thread(_call_image_model)


# =========================
# 7) Discord 이벤트
# =========================
@client.event
async def on_ready():
    await tree.sync()
    print(f"✅ 로그인 완료: {client.user} (ID: {client.user.id})")
    print("✅ 슬래시 커맨드 동기화 완료")
    print("✅ 이제 @멘션 대화 / 그림 요청 가능")


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
            result = await generate_rika_response(
                user_display_name=user_display_name,
                user_message=user_text,
                history=history
            )

        reply_text = result["reply"]
        mode = result["mode"]

        # 대화 메모리 저장
        history.append({"role": "user", "text": user_text})
        history.append({"role": "rika", "text": reply_text})

        if mode == "image":
            image_prompt = result["image_prompt"]

            if not image_prompt:
                await message.channel.send(
                    f"{reply_text}\n\n(근데 이미지 프롬프트가 비어 있어서 그림 생성은 못 했어.)"
                )
                return

            async with message.channel.typing():
                generated_path = await generate_image_with_gemini(image_prompt)

            file = discord.File(generated_path, filename=os.path.basename(generated_path))
            await message.channel.send(content=reply_text, file=file)
            return

        # mode == "chat"
        emotion = result["emotion"]
        image_path = get_static_image_path(emotion)

        if image_path:
            file = discord.File(image_path, filename=os.path.basename(image_path))
            await message.channel.send(content=reply_text, file=file)
        else:
            await message.channel.send(
                content=f"{reply_text}\n\n(참고: `{emotion}` 이미지 파일을 찾지 못했어.)"
            )

    except json.JSONDecodeError:
        await message.channel.send("우우... 답변 형식이 조금 꼬였어. 한 번만 다시 말해줘!")
    except Exception as e:
        await message.channel.send(f"아와와... 오류가 났어.\n`{type(e).__name__}: {e}`")


# =========================
# 8) 보조 슬래시 명령어
# =========================
@tree.command(name="reset", description="최근 대화 기억 초기화")
async def reset_memory(interaction: discord.Interaction):
    user_id = interaction.user.id
    user_memories[user_id].clear()
    await interaction.response.send_message(
        "오케이. 방금까지의 대화 기억은 잠깐 리셋해뒀어.",
        ephemeral=True
    )


@tree.command(name="ping", description="봇이 살아있는지 확인")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("퐁. 아마미야 리카, 정상 작동 중이야.", ephemeral=True)


# =========================
# 9) 실행
# =========================
client.run(DISCORD_BOT_TOKEN)