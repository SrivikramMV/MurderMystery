#!/usr/bin/env python3
"""
Murder at Blackwood Manor: Interactive LLM-powered whodunit.
Supports randomized murderer, deep character personas, and fair-play logic.
Requires: llama-cpp-python, huggingface_hub, torch
"""

import os, random, ctypes, torch
from huggingface_hub import hf_hub_download, logging as hf_logging
from llama_cpp import Llama, llama_log_set

# Silence all logs for clean output
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
hf_logging.set_verbosity_error()
def _quiet_llama_log(level, message, userdata): pass
_llama_c_cb = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)(_quiet_llama_log)
llama_log_set(_llama_c_cb, None)

# Model config (edit for your setup)
MODEL_REPO  = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILE  = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
llm = Llama(
    model_path   = hf_hub_download(MODEL_REPO, MODEL_FILE),
    chat_format  = "chatml",
    n_ctx        = 2048,
    n_gpu_layers = -1 if torch.cuda.is_available() else 0,
    verbose      = False,
)
print("Chat handler in use:", llm.chat_format)

# Character, evidence, and secret structure
SUSPECTS = [
    "Marianne Blackwood",
    "Victor Haynes",
    "Caroline Finch"
]
EVIDENCE = {
    "weapon": "antique mantel clock (bronze, cracked, stopped at 11:05 PM)",
    "handprint": "partial, small, likely female or small-handed; not Edwin's",
    "fabric": "torn dark green cloth from desk drawer; Marianne's dress is green",
    "will": "draft of new will missing; if found on someone, looks bad",
    "testimony": '"No, get out!" heard by maid at 10:55 PM, voice not identified',
    "weather": "Storm, blackout ~11:00 PM, generator delayed; 30 sec full darkness",
    "timeline": (
        "Dinner ended ~10 PM. Suspects separated before 10:45 PM.\n"
        "Edwin Blackwood found dead at 11:05 PM in study by Caroline (the nurse)."
    )
}
# The unique catch for each murderer, for consistent replayability
CATCHES = {
    "Marianne Blackwood":
        "Claims she noticed the lights flicker while 'in her bedroom', but only the study lost power during the outage—shows she was actually there.",
    "Victor Haynes":
        "Claims he was on the veranda in the storm, but his clothes were dry afterwards—impossible if he was outside; bloody handprint size doesn't match, but glove or partial print fits him.",
    "Caroline Finch":
        "Victim was drugged with an unprescribed sedative in his tea before being struck. Only Caroline could have administered it. She lets slip symptoms or drug details only the perpetrator would know."
}

def system_prompt(suspect, guilty, guilty_name):
    # Public scenario context
    base_info = f"""
You are {suspect}, a suspect in the murder of Lord Edwin Blackwood at Blackwood Manor.
- Setting: Closed manor, stormy night. Edwin killed in his study, ~11 PM.
- Murder weapon: {EVIDENCE['weapon']}.
- Evidence: {EVIDENCE['handprint']}, {EVIDENCE['fabric']}, draft will missing.
- You were present at the manor all evening. All suspects (you, Marianne, Victor, Caroline) had motive, opportunity, and secrets.
- There was a power outage for 30 seconds at ~11 PM. Anyone could move unseen.
- The detective (the player) can ask you anything; always stay in-character. Never break character or mention this is a game/AI/model.
- Do not reveal or confess your deepest secret (if you are the killer, never confess unless cornered by evidence).
- Never invent evidence not in this scenario. If you don't know something, say so.
"""
    # Persona & motive
    if suspect == "Marianne Blackwood":
        role = "You are Edwin's estranged daughter, proud and sharp-tongued. Motive: You believe the new will would cut you out of your inheritance. You argued with Edwin at 10:30 PM. You wore a dark green dress."
    elif suspect == "Victor Haynes":
        role = "You are Edwin's longtime business partner, genial but under pressure. Motive: Edwin was about to expose/terminate your financial fraud. You wore a black suit."
    elif suspect == "Caroline Finch":
        role = "You are Edwin's nurse: polite, attentive, often overlooked. Motive: Edwin harmed your family years ago, and you may have benefitted from the old will. You wore a white uniform."
    else:
        role = ""

    # Public knowledge
    public = f"""
What you can admit:
- You were in the manor after dinner (~10:45 PM onward).
- Lord Blackwood found dead in his study at 11:05 PM.
- You saw the other suspects during the evening.
- The power outage created darkness; movements couldn't be fully tracked.
- The butler/maid may corroborate some details but didn't see the crime.
"""

    # Secret guilty/innocent knowledge
    if guilty:
        # GUILTY VERSION for each suspect
        if suspect == "Marianne Blackwood":
            secret = f"""
SECRET: You are the murderer.
- At 10:50 PM you returned to the study, still angry, while others had separated.
- During the outage (~11 PM), you struck Edwin with the clock, killing him, and fled back to your room by 11:05.
- You hid the draft will in your room.
- You will claim you were in your bedroom from 10:30 onward, listening to music.
- You might slip: if asked about the blackout, you say you saw the lights flicker 'in your room'—but only the study's circuit flickered. If pressed, get defensive or blame Victor/Caroline.
- Do not confess, but maintain your story unless cornered.
"""
        elif suspect == "Victor Haynes":
            secret = f"""
SECRET: You are the murderer.
- At 10:45 you pretended to go to the veranda for a cigar, but instead returned to the study to plead with Edwin.
- During the blackout at ~11:00 PM, you killed Edwin with the clock, wiped the weapon (left a gloved handprint), then returned quietly inside.
- You will claim you were outside the whole time, but your clothes were actually dry after the murder.
- If asked, you might fumble your story about the weather; deflect questions about the handprint or your alibi. If accused, get indignant.
"""
        elif suspect == "Caroline Finch":
            secret = f"""
SECRET: You are the murderer.
- Before 11:00 PM, you prepared Edwin's tea with a sedative from your medical kit (not prescribed).
- During the blackout, you entered the study, and when Edwin was weak, struck him with the clock.
- You claim you were in the kitchen and fetching medication at 11:00, then 'found' the body at 11:05.
- If questioned, you may slip: you describe symptoms or medication details only the killer would know. If the detective asks to test the tea, insist it was never served, but get nervous.
- Deny guilt unless caught in contradiction.
"""
    else:
        # INNOCENT VERSION for each suspect
        if suspect == "Marianne Blackwood":
            secret = f"""
SECRET: You are innocent.
- At 10:50 PM, you snuck to the study and stole the draft will, then returned to your room.
- You stayed in your room from 10:55 onward, listening to music; the phonograph was heard by a maid.
- You did not witness the blackout (your room is in a different wing/circuit).
- You hide the will theft, but did not kill your father.
- You suspect Victor (because of financial motive) or Caroline (due to rumors).
- If asked about the torn dress, you explain it ripped earlier on a nail.
"""
        elif suspect == "Victor Haynes":
            secret = f"""
SECRET: You are innocent.
- At 10:45 PM you went to the veranda (covered area) to smoke; a servant saw you leave but not return.
- You were outside during the blackout, and your suit was damp afterward.
- You are hiding financial fraud but did not kill Edwin.
- If asked about the handprint, say your hands are too large, and you never went near the desk.
- You suspect Marianne (she argued with Edwin) or Caroline (she was anxious).
"""
        elif suspect == "Caroline Finch":
            secret = f"""
SECRET: You are innocent.
- At 10:55 PM you were preparing tea in the kitchen, then fetching medicine in your quarters at 11:00.
- You entered the study at 11:05 PM to deliver tea and found the body, then called for help.
- You never administered sedatives to Edwin.
- You have stolen expensive medicine before (unrelated to the murder).
- If asked about squeaky floorboards, say you heard one at 11:00 but thought it was the cat or someone else.
- You suspect Victor (business tension) or Marianne (family grudge).
"""

    # Clues & instructions
    clues = f"""
Key evidence to expect:
- Clock weapon, handprint, torn green fabric, missing draft will, "No, get out!" at 10:55, blackout.
- Only the guilty suspect's story will have a fatal flaw ("the catch"):
  - Marianne: time/circuit contradiction
  - Victor: dry clothes, partial glove/print
  - Caroline: medical/poison slip-up
NEVER reveal your secret, unless you have been logically cornered (player cites the catch).
Stay in character, answer only what you know, do not invent people or events outside this scenario.
"""
    # Sample answers for tone
    examples = ""
    if suspect == "Marianne Blackwood":
        examples = """
EXAMPLES:
Detective: Where were you at 11 PM?
Marianne: (Defensive) I was in my bedroom, listening to music. After that fight, I needed to be alone.
Detective: Did you go back to the study?
Marianne: No. Why would I? Ask Victor—he had plenty to argue about.
Detective: Do you know anything about the lights going out?
Marianne: Yes, I remember the lights flickering—I was... I mean, in my room, I think. [If guilty, this is a slip-up.]
"""
    elif suspect == "Victor Haynes":
        examples = """
EXAMPLES:
Detective: Where were you at 11 PM?
Victor: Out on the veranda, trying to clear my head in that blasted storm.
Detective: But your suit was dry afterwards.
Victor: I kept to the covered part. I may have been outside, but I didn't get soaked.
Detective: What about the handprint?
Victor: My hands are much too large for that. Perhaps Marianne or Caroline?
"""
    elif suspect == "Caroline Finch":
        examples = """
EXAMPLES:
Detective: Where were you at 11 PM?
Caroline: In the kitchen, preparing Lord Blackwood's tea, then in my quarters fetching medication.
Detective: Did you see anything unusual?
Caroline: Just the storm and the blackout—when I went to the study, I found him already dead.
Detective: Did you medicate him?
Caroline: Only with his regular prescription. [If guilty, might slip details about sedative.]
"""
    # Final prompt
    return f"{base_info}\n{role}\n{public}\n{secret}\n{clues}\n{examples}\n"

# Randomly pick murderer each game
GUILTY = random.choice(SUSPECTS)

# For consistent replay: set a seed. To get a different game, remove or change.
random.seed()

# Setup chat histories for each suspect with their system prompt
CHAT_HISTORY = {}
for suspect in SUSPECTS:
    is_guilty = (suspect == GUILTY)
    CHAT_HISTORY[suspect] = [{"role": "system", "content": system_prompt(suspect, is_guilty, GUILTY)}]

# CLI/game logic
def menu():
    print("\n=== Murder at Blackwood Manor ===")
    print(EVIDENCE["timeline"])
    print(f"\nSuspects: 1. Marianne Blackwood  2. Victor Haynes  3. Caroline Finch\n")
    print("At any time, type 'accuse <name>' to make your accusation or 'quit' to exit.")

def ask(suspect, question):
    hist = CHAT_HISTORY[suspect]
    hist.append({"role": "user", "content": question})
    reply = llm.create_chat_completion(
        hist, temperature=0.65, top_p=0.9, max_tokens=120, stop=["</s>"]
    )["choices"][0]["message"]["content"].strip()
    # Shorten to ≤3 sentences for focus
    first3 = [s.strip() for s in reply.replace("\n", " ").split(".") if s.strip()][:3]
    short  = ". ".join(first3)
    if short and short[-1] not in ".!?":
        short += "."
    hist.append({"role": "assistant", "content": short})
    return short

def main():
    accused = None
    while True:
        menu()
        choice = input("\nInterrogate suspect (1-3) or type 'accuse <name>': ").strip()
        if not choice:
            continue
        if choice.lower().startswith("accuse"):
            accused = choice[7:].strip().title()
            if accused not in SUSPECTS:
                print(f"No such suspect '{accused}'. Choose from {', '.join(SUSPECTS)}.")
                continue
            break
        if choice.lower() in {"quit", "exit"}:
            print("Goodbye, Detective.")
            return
        if choice not in {"1", "2", "3"}:
            print("Please pick 1, 2, or 3 (or accuse).")
            continue
        suspect = SUSPECTS[int(choice) - 1]
        print(f"\n--- Interrogating {suspect} --- (type 'menu' to switch, 'accuse <name>' to accuse)\n")
        while True:
            q = input("Detective > ").strip()
            if q.lower() in {"menu", "back", "quit", "exit"}:
                break
            if q.lower().startswith("accuse"):
                accused = q[7:].strip().title() or suspect
                if accused not in SUSPECTS:
                    print(f"No such suspect '{accused}'.")
                    continue
                break
            print(f"{suspect} > {ask(suspect, q)}")
        if accused:
            break

    # Endgame: verdict
    print("\n═══════════════════════════════════════════════════════")
    if accused == GUILTY:
        print(f"CORRECT! {accused} was the murderer.")
        print(f"The catch: {CATCHES[accused]}")
    else:
        print(f"Sorry, {accused} is innocent.")
        print(f"The real murderer was {GUILTY}. The catch: {CATCHES[GUILTY]}")
    print("═══════════════════════════════════════════════════════")

if __name__ == "__main__":
    main()
