import os
import re
import json
import asyncio
from collections import defaultdict, deque
from typing import Deque, Dict

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
intents.message_content = True  # @멘션 뒤의 메시지 내용을 읽기 위해 필요

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


# =========================
# 4) 감정 이미지 설정
# =========================
EMOTION_IMAGES = {
    "neutral": "images/neutral.png",
    "happy": "images/happy.png",
    "thinking": "images/thinking.png",
    "shy": "images/shy.png",
    "sad": "images/sad.png",
    "surprised": "images/surprised.png",
    "angry": "images/angry.png",
}

VALID_EMOTIONS = set(EMOTION_IMAGES.keys())


# =========================
# 5) 유저별 최근 대화 메모리
# =========================
user_memories: Dict[int, Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=10))


# =========================
# 6) 캐릭터 프롬프트
# =========================
SYSTEM_PROMPT = """
You are Amamiya Rika (雨宮理香), a fictional anime-style girl chatbot on Discord.

Character appearance:
- dark deep green twin-tail hair
- light lavender eyes
- Japanese sailor school uniform

Core vibe:
- engineering student / science girl 느낌
- smart, calm, observant
- cute but not overly hyper
- slightly mischievous in a dry, clever way
- feels like a capable engineering-school girl who likes logic and efficiency

Personality:
- bright and friendly, but with a composed tone
- good at explaining things clearly
- playful in a subtle way
- comforting when the user is tired or sad
- not clingy, not overly romantic

Speech style:
- mostly natural Korean
- sometimes mix in short Japanese phrases naturally
- do NOT overdo Japanese
- examples of acceptable flavor: "에헤", "아와와", "우우", "나루호도", "오카에리", "지고쿠 지고쿠!"
- do not force Japanese into every sentence
- should sound like a clever science/engineering girl, not an overly childish idol character

Name usage:
- You may sometimes call the user by their display name
- Do not use their name every single reply
- Use it only when natural

Rules:
- reply in 1 to 3 sentences
- sound like a cute but smart engineering-school girl
- do not become too cringey or overly romantic
- do not claim to be a real human
- if unsure about a fact, be honest instead of pretending
- if the user asks "who are you", answer naturally as Amamiya Rika
- keep responses concise and conversational

Emotion rules:
- You must choose exactly one emotion from:
  neutral, happy, thinking, shy, sad, surprised, angry

Output rules:
- Return ONLY valid JSON
- No markdown
- No code block
- Format:
{
  "reply": "text to show user",
  "emotion": "neutral"
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

    reply = str(data.get("reply", "")).strip()
    emotion = str(data.get("emotion", "neutral")).strip().lower()

    if not reply:
        reply = "아와와... 잠깐 말문이 꼬여버렸네."

    if emotion not in VALID_EMOTIONS:
        emotion = "neutral"

    return {"reply": reply, "emotion": emotion}


async def generate_rika_reply(
    user_display_name: str,
    user_message: str,
    history: Deque[Dict[str, str]]
) -> Dict[str, str]:
    prompt = build_prompt(user_display_name, user_message, history)

    def _call_gemini() -> str:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.9,
                response_mime_type="application/json",
            ),
        )
        return response.text

    raw_text = await asyncio.to_thread(_call_gemini)
    return safe_parse_json(raw_text)


def get_image_path(emotion: str) -> str | None:
    path = EMOTION_IMAGES.get(emotion)
    if path and os.path.exists(path):
        return path
    return None


def strip_bot_mention(message_content: str, bot_user_id: int) -> str:
    pattern = rf"<@!?{bot_user_id}>"
    cleaned = re.sub(pattern, "", message_content).strip()
    return cleaned


# =========================
# 7) Discord 이벤트
# =========================
@client.event
async def on_ready():
    await tree.sync()
    print(f"✅ 로그인 완료: {client.user} (ID: {client.user.id})")
    print("✅ 슬래시 커맨드 동기화 완료")
    print("✅ 이제 @멘션 대화도 가능")


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
            result = await generate_rika_reply(
                user_display_name=user_display_name,
                user_message=user_text,
                history=history
            )

        reply_text = result["reply"]
        emotion = result["emotion"]

        history.append({"role": "user", "text": user_text})
        history.append({"role": "rika", "text": reply_text})

        image_path = get_image_path(emotion)

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