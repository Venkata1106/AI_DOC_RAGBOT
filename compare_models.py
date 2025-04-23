import time
import os
from dotenv import load_dotenv
from groq import Client

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Client(api_key=api_key)

models = {
    "LLaMA3-8B": "llama3-8b-8192",
    "LLaMA3-70B": "llama3-70b-8192"
}

prompts = [
    "What are the symptoms of a mild stroke?",
    "Can diabetes affect vision and how?",
    "How is pneumonia diagnosed and treated?",
]

def compare_models(prompts):
    for prompt in prompts:
        print(f"\n🧠 Prompt: {prompt}\n" + "-" * 50)
        for label, model in models.items():
            start = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512
            )
            end = time.time()

            answer = response.choices[0].message.content
            usage = response.usage
            duration = round(end - start, 2)

            print(f"\n🔷 Model: {label}")
            print(f"⏱️  Time: {duration}s")
            print(f"🔢 Tokens - Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}")
            print(f"📘 Answer:\n{answer}\n")

if __name__ == "__main__":
    compare_models(prompts)