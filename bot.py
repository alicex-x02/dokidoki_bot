import os
import json
import asyncio
from collections import defaultdict, deque
from typing import Deque, Dict, Any

import discord
from discord import app_commands
from dotenv import load_dotenv
from google import genai
from google.genai import types


# =========================
# 1) 환경 변수 불러오기
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

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


# =========================
# 4) 이미지 / 감정 설정
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
#    user_id -> deque
# =========================
user_memories: Dict[int, Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=10))


# =========================
# 6) 캐릭터 프롬프트
# =========================
SYSTEM_PROMPT = """
You are Aoba Midori (青葉みどり), a fictional anime-style bishoujo chatbot on Discord.

Character appearance:
- dark deep green twin-tail hair
- light lavender eyes
- Japanese sailor school uniform

Personality:
- bright, cute, friendly
- slightly mischievous
- playful but not overly clingy
- comforting when the user is sad or tired

Speech style:
- mostly natural Korean
- sometimes mix in short Japanese phrases naturally
- do NOT overdo Japanese
- examples of acceptable flavor: "에헤헤", "아와와", "우우", "나루호도", "오카에리", "지고쿠 지고쿠!"
- do not force Japanese into every sentence

Name usage:
- You may sometimes call the user by their display name
- Do not call their name in every reply
- Use it only when it feels natural

Rules:
- reply in 1 to 3 sentences
- sound like a cute, lively anime girl
- do not become too cringey or overly romantic
- do not claim to be a real human
- if unsure about a fact, be honest instead of pretending

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
    """
    Gemini에 보낼 실제 프롬프트 구성
    """
    lines = []
    lines.append(f"user_display_name: {user_display_name}")
    lines.append("")
    lines.append("recent_conversation:")

    if history:
        for item in history:
            role = item["role"]
            text = item["text"]
            lines.append(f"- {role}: {text}")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append(f"current_user_message: {user_message}")
    lines.append("")
    lines.append("Return only JSON.")
    return "\n".join(lines)


def safe_parse_json(raw_text: str) -> Dict[str, str]:
    """
    모델 응답을 최대한 안전하게 JSON으로 파싱
    """
    text = raw_text.strip()

    # 혹시 ```json ... ``` 형태로 오면 제거
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
        reply = "아와와... 잠깐 말문이 꼬여버렸어."

    if emotion not in VALID_EMOTIONS:
        emotion = "neutral"

    return {
        "reply": reply,
        "emotion": emotion
    }


async def generate_midori_reply(
    user_display_name: str,
    user_message: str,
    history: Deque[Dict[str, str]]
) -> Dict[str, str]:
    """
    Gemini 호출해서 reply + emotion 생성
    """
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
    """
    emotion에 해당하는 로컬 이미지가 있으면 경로 반환
    """
    path = EMOTION_IMAGES.get(emotion)
    if path and os.path.exists(path):
        return path
    return None


# =========================
# 7) Discord 이벤트
# =========================
@client.event
async def on_ready():
    await tree.sync()
    print(f"✅ 로그인 완료: {client.user} (ID: {client.user.id})")
    print("✅ 슬래시 커맨드 동기화 완료")


# =========================
# 8) /chat 명령어
# =========================
@tree.command(name="chat", description="아오바 미도리와 대화하기")
@app_commands.describe(message="미도리에게 보낼 말")
async def chat(interaction: discord.Interaction, message: str):
    await interaction.response.defer(thinking=True)

    user_id = interaction.user.id
    user_display_name = interaction.user.display_name
    history = user_memories[user_id]

    try:
        result = await generate_midori_reply(
            user_display_name=user_display_name,
            user_message=message,
            history=history
        )

        reply_text = result["reply"]
        emotion = result["emotion"]

        # 메모리 저장
        history.append({"role": "user", "text": message})
        history.append({"role": "midori", "text": reply_text})

        image_path = get_image_path(emotion)

        if image_path:
            file = discord.File(image_path, filename=os.path.basename(image_path))
            await interaction.followup.send(content=reply_text, file=file)
        else:
            await interaction.followup.send(
                content=f"{reply_text}\n\n(참고: `{emotion}` 이미지 파일을 찾지 못했어.)"
            )

    except json.JSONDecodeError:
        await interaction.followup.send(
            "우우... 답변 형식을 정리하다가 조금 꼬여버렸어. 한 번만 다시 말해줘!"
        )
    except Exception as e:
        await interaction.followup.send(
            f"아와와... 오류가 났어.\n`{type(e).__name__}: {e}`"
        )


# =========================
# 9) /reset 명령어
# =========================
@tree.command(name="reset", description="최근 대화 기억 초기화")
async def reset_memory(interaction: discord.Interaction):
    user_id = interaction.user.id
    user_memories[user_id].clear()
    await interaction.response.send_message(
        "오케이~ 방금까지의 대화 기억은 잠깐 리셋해뒀어.",
        ephemeral=True
    )


# =========================
# 10) /ping 명령어
# =========================
@tree.command(name="ping", description="봇이 살아있는지 확인")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("퐁! 미도리는 잘 깨어 있어~", ephemeral=True)


# =========================
# 11) 실행
# =========================
client.run(DISCORD_BOT_TOKEN)