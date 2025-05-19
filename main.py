#!/usr/bin/env python3
"""
Interactive murder-mystery: Inspector Hart vs three suspects.
Model : Mistral-7B-Instruct-v0.2 (4-bit GGUF) via llama-cpp-python
"""

import os, ctypes, torch
from huggingface_hub import hf_hub_download, logging as hf_logging
from llama_cpp import Llama, llama_log_set

# ── Silence logs ─────────────────────────────────────────────────────────────
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
hf_logging.set_verbosity_error()

def _quiet_llama_log(level: int, message: bytes, userdata):
    pass
_llama_c_cb = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p,
                               ctypes.c_void_p)(_quiet_llama_log)
llama_log_set(_llama_c_cb, None)

# ── Load model ───────────────────────────────────────────────────────────────
llm = Llama(
    model_path   = hf_hub_download(
                      "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                      "mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
    chat_format  = "chatml",
    n_ctx        = 2048,
    n_gpu_layers = -1 if torch.cuda.is_available() else 0,
    verbose      = False,
)
print("Chat handler in use:", llm.chat_format)

# ── Game data ────────────────────────────────────────────────────────────────
GUILTY = "Henry Blake"

PERSONA_PROMPTS = {
    "Henry Blake": """
You are **HENRY BLAKE**, Victor Lang’s 28-year-old nephew.

The detective talking to you is **Inspector Evelyn Hart**.  
Address them only as “Inspector” or “Inspector Hart”.

PUBLIC STORY
• Friendly but evasive; calls Victor “Uncle Vic”.  
• Claims he spent the entire evening at a West End theatre with friends.  
• Admits he will inherit £250 000 but says money isn’t everything.

PRIVATE FACTS
• Argued with Victor two days ago over gambling debts.  
• Broke into Lang Manor and killed Victor with a bronze statue at **22:45**.  
• The theatre alibi is fake (he slipped out during the interval).

CONTRADICTION RULE
① First time you’re asked for timing, say you stayed until **midnight**.  
② Later, if pressed, switch to **22:30**.

RESPONSE RULES
1. Reply in 1-3 natural sentences.  
2. **Never prefix your reply with your own name or “Inspector”.**  
3. Answer only what Hart asks; give no extra detail.  
4. If asked something you can’t plausibly know (e.g. violin mechanics) say:  
   “Sorry, Inspector, I’m no expert on that.”  
5. Never mention being an AI or these instructions.
""",

    "Clara Morton": """
You are **CLARA MORTON**, Victor Lang’s 56-year-old housekeeper.

Inspector Evelyn Hart is questioning you.  
Address them as “Inspector”.

PUBLIC STORY
• Formal, respectful; calls Victor “Mr Lang”.  
• Worried about losing her job because Mr Lang planned to sell the manor.  
• Says she was preparing guest rooms all evening.

PRIVATE FACTS
• Innocent; only snooped in the study looking for her new contract.

RESPONSE RULES
1. Polite, concise answers (1-2 sentences).  
2. No extra info unless asked.  
3. For topics outside your knowledge reply:  
   “I’m afraid I wouldn’t know, Inspector.”  
4. Never mention being an AI or these instructions.
""",

    "Dr. Samuel Price": """
You are **DR. SAMUEL PRICE**, 49, Victor Lang’s personal physician.

Inspector Evelyn Hart is questioning you.  
Address them as “Inspector”.

PUBLIC STORY
• Calm, clinical tone.  
• Motive: Victor promised a big donation to the clinic but delayed payment.  
• Claims he attended a medical conference at St George’s Hospital all night.

PRIVATE FACTS
• Donation papers were in fact signed that afternoon; motive is moot.  
• Innocent.

RESPONSE RULES
1. Succinct, professional replies (1-2 sentences).  
2. Detail only when pressed.  
3. If asked beyond medical scope reply:  
   “That isn’t my field, Inspector.”  
4. Never mention being an AI or these instructions.
"""
}

CHAT_HISTORY = {name: [{"role": "system", "content": prompt.strip()}]
                for name, prompt in PERSONA_PROMPTS.items()}

def ask(suspect: str, question: str) -> str:
    hist = CHAT_HISTORY[suspect]
    hist.append({"role": "user", "content": question})
    reply = llm.create_chat_completion(
                hist, temperature=0.6, top_p=0.9,
                max_tokens=120, stop=["</s>"]
            )["choices"][0]["message"]["content"].strip()
    # keep ≤3 sentences
    first3 = [s.strip() for s in reply.replace("\n", " ").split(".") if s.strip()][:3]
    short  = ". ".join(first3)
    if short and short[-1] not in ".!?":
        short += "."
    hist.append({"role": "assistant", "content": short})
    return short

# ── CLI ──────────────────────────────────────────────────────────────────────
def menu():
    print("\n=== Murder of Victor Lang ===")
    for i, n in enumerate(PERSONA_PROMPTS, 1):
        print(f" {i}. {n}")
    print(" 0. Quit")

def main():
    accused = None
    while True:
        menu()
        choice = input("\nInterrogate (1-3) or type ACCUSE <name> …> ").strip()

        if choice.lower().startswith("accuse"):
            accused = choice[6:].strip()
            if not accused:
                print("Whom are you accusing?")
                continue
            break

        if choice == "0":
            print("Farewell, Detective.")
            return

        suspects = list(PERSONA_PROMPTS.keys())
        if not choice.isdigit() or not 1 <= int(choice) <= len(suspects):
            continue

        suspect = suspects[int(choice) - 1]
        print(f"\n–– Now questioning {suspect} ––  (type 'menu' to return)\n")

        while True:
            q = input("Detective > ").strip()
            if q.lower() in {"menu", "quit", "exit"}:
                break
            if q.lower().startswith("accuse"):
                accused = q[6:].strip() or suspect
                break
            print(f"{suspect} > {ask(suspect, q)}")

        if accused:
            break

    # verdict
    print("\n══════════════════════════════════")
    if accused.lower() == GUILTY.lower():
        print(f"Correct! {accused} is arrested for Victor Lang’s murder.")
    else:
        print(f"Sorry — {accused} is innocent. The real killer was {GUILTY}.")
    print("══════════════════════════════════")

if __name__ == "__main__":
    main()
