# Forum AI — Project Context for Claude Code

## Mục tiêu dự án
Ứng dụng **forum chat** nơi nhiều AI agent thảo luận về một topic, sau đó tổng hợp kết luận.

## Môi trường
- **OS:** Windows, thư mục dự án tại `C:\Users\Administrator\Dropbox\forum_ai`
- **Python:** 3.13, virtual env tại `venv\`
- **Kích hoạt venv:** `venv\Scripts\activate`

## Cấu trúc thư mục
```
forum_ai/
├── agents/
│   ├── groq_agent.py       # Llama3 via Groq (FREE)
│   ├── gemini_agent.py     # Gemini 2.5 Flash Lite (FREE)
│   ├── mistral_agent.py    # Mistral Large (FREE)
│   ├── gpt_agent.py        # ChatGPT — chưa dùng (cần trả phí)
│   ├── deepseek_agent.py   # DeepSeek — chưa dùng (hết balance)
│   └── claude_agent.py     # Claude — chưa implement
├── core/
│   ├── base_agent.py       # Abstract base class cho tất cả agents
│   ├── history.py          # Quản lý lịch sử hội thoại
│   ├── moderator.py        # Điều phối lượt nói, vòng thảo luận
│   └── synthesizer.py      # Tổng hợp kết luận (dùng Mistral)
├── config/
│   └── settings.py         # ForumConfig, DEFAULT_CONFIG, round_prompts
├── utils/
│   └── display.py          # In màu terminal
├── main.py                 # CLI — đang chạy ổn định
├── app.py                  # Streamlit UI — CHƯA CÓ, cần tạo
├── test_keys.py            # Kiểm tra API keys
├── requirements.txt
└── .env                    # API keys (không commit)
```

## API Keys cần có trong .env
```
GROQ_API_KEY=...        # console.groq.com — FREE
GOOGLE_API_KEY=...      # aistudio.google.com — FREE
MISTRAL_API_KEY=...     # console.mistral.ai — FREE
# OPENAI_API_KEY=...    # Chưa dùng
# DEEPSEEK_API_KEY=...  # Chưa dùng
```

## Agents đang hoạt động

### GroqAgent (Llama3)
- Model: `llama-3.3-70b-versatile`
- API: OpenAI-compatible qua `https://api.groq.com/openai/v1`
- Persona: Stubborn, data-driven, không đổi quan điểm trừ khi có bằng chứng cứng
- Fix đã làm: build_turn_instruction override để tránh bịa số liệu

### GeminiAgent
- Model: `gemini-2.5-flash-lite` (dùng model này vì free, ít bị safety filter)
- API: `google-genai` SDK, `genai.Client`
- Fix đã làm: retry 3 lần khi 503, safety_settings BLOCK_ONLY_HIGH, max_output_tokens=1000

### MistralAgent
- Model: `mistral-large-latest`
- API: `mistralai` SDK
- max_tokens=1000
- Cũng dùng làm Synthesizer

## Core Logic

### ForumConfig (config/settings.py)
```python
max_turns: int = 9          # 9 turns = 3 agents x 3 vòng
max_context_messages: int = 20
temperature: float = 0.75
max_tokens_per_turn: int = 500
round_prompts: tuple = (Opening, Debate, Rebuttal, Closing)
```

### Round structure (9 turns, 3 agents)
| Turn | Agent   | Round    |
|------|---------|----------|
| 1    | Llama3  | Opening  |
| 2    | Gemini  | Opening  |
| 3    | Mistral | Opening  |
| 4    | Llama3  | Debate   |
| 5    | Gemini  | Debate   |
| 6    | Mistral | Debate   |
| 7    | Llama3  | Closing  |
| 8    | Gemini  | Closing  |
| 9    | Mistral | Closing  |

### history.format_for_openai()
Tất cả messages đều được format thành role="user" với prefix `[AgentName said]: ...`
→ Tránh lỗi Mistral tự tranh luận với chính mình

## Bugs đã fix
1. Gemini bị cắt giữa câu → đổi sang gemini-2.5-flash-lite, tăng max_output_tokens=1000
2. Gemini 503 crash → retry 3 lần với sleep 10-20s
3. Mistral tự tranh luận với chính mình → fix format_for_openai() mapping tất cả về role=user
4. Llama3 bịa số liệu → thêm anti-hallucination instruction trong build_turn_instruction
5. IndentationError khi paste code → cần xóa sạch file trước khi paste (Clear-Content trong PowerShell)

## Việc cần làm tiếp — CÔNG VIỆC HIỆN TẠI
**Tạo file `app.py` — Streamlit UI** với các tính năng:
- Sidebar: chỉnh max_turns, temperature, max_tokens
- Ô nhập topic
- Streaming từng câu trả lời theo thời gian thực (word-by-word)
- Progress bar
- Màu sắc riêng cho từng agent
- Phần synthesis ở cuối
- Nút download transcript
- Chạy bằng: `streamlit run app.py`

## Thư viện đã cài
```
openai
google-genai
mistralai
python-dotenv
streamlit>=1.35.0
```

## Lệnh chạy
```bash
# CLI
python main.py

# Streamlit (chưa có app.py)
streamlit run app.py

# Test API keys
python test_keys.py
```
