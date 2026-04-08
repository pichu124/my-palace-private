from dotenv import load_dotenv
load_dotenv()
import os

def test_groq():
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1"
    )
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say 'Groq OK' only."}],
        max_tokens=10
    )
    print("✅ Groq (Llama3):", r.choices[0].message.content.strip())

def test_gemini():
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    r = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say 'Gemini OK' only.",
        config=types.GenerateContentConfig(max_output_tokens=10)
    )
    print("✅ Gemini:", r.text.strip())

def test_mistral():
    from mistralai import Mistral
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    r = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": "Say 'Mistral OK' only."}],
        max_tokens=10
    )
    print("✅ Mistral:", r.choices[0].message.content.strip())

def test_deepseek():
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1"
    )
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Say 'DeepSeek OK' only."}],
        max_tokens=10
    )
    print("✅ DeepSeek:", r.choices[0].message.content.strip())

tests = [
    ("Groq",     test_groq),
    ("Gemini",   test_gemini),
    ("Mistral",  test_mistral),
    ("DeepSeek", test_deepseek),
]

for name, fn in tests:
    try:
        fn()
    except Exception as e:
        print(f"❌ {name} FAILED: {e}")