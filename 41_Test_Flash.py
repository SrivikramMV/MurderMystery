#!/usr/bin/env python3
"""
Murder at Blackwood Manor – Gemini 2.5 Flash API Edition
Randomizes murderer, robust persona prompts for deep roleplay.
Author: Dragon & ChatGPT
"""

import os, random, requests

GEMINI_API_KEY = "AIzaSyBYIm2tPFL93wHZRUGuF5PoNdGok01uqhI"
GEMINI_MODEL = "gemini-2.5-flash-preview-04-17"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
# GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

def gemini_generate(prompt):
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    resp = requests.post(GEMINI_URL, json=payload, timeout=60)
    if not resp.ok:
        print("ERROR: Gemini API request failed:", resp.text)
        return "Sorry, I can't respond right now."
    try:
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return "Sorry, something went wrong with the AI response."

def format_query_for_llm(suspect, question):
    system_msg = (
        f"You are {suspect}, being interrogated at Blackwood Manor by Inspector Hart, the detective. "
        f"Inspector Hart is asking you questions about the murder of Lord Edwin Blackwood. "
        "You must answer ONLY as yourself (never as the detective, never as a narrator, never as Lord Edwin)."
    )
    return f"{system_msg}\nDetective: {question}"

SUSPECTS = ["Marianne Blackwood", "Victor Haynes", "Caroline Finch"]
VICTIM = "Lord Edwin Blackwood"
SCENE_DESCRIPTION = (
    "It is the morning after a stormy night at Blackwood Manor. Lord Edwin Blackwood, "
    "aged 68, head of the Blackwood family, was found dead in his study, bludgeoned with "
    "an antique bronze clock. All present were in the manor at the time of the murder (approx. 11:00 PM). "
    "The manor was locked down due to the storm—a closed-circle mystery."
)
STARTING_CLUES = [
    "- Murder weapon: Antique mantel clock, found cracked on the floor. Hands stopped at 11:05 PM.",
    "- Partial bloody handprint (smaller than Edwin's) on the desk.",
    "- Torn bit of dark green cloth caught on a desk drawer.",
    "- The draft of a new will is missing from the study.",
    '- "No, get out!" was shouted by the victim at 10:55 PM, heard by a housemaid.',
    "- The power flickered out at 11:00 PM, restoring 30 seconds later. All suspects knew the layout well.",
    "- A severe thunderstorm was raging throughout.",
    "- Caroline found the body at 11:05 PM, bringing Lord Blackwood's nightly herbal tea."
]

def build_system_prompt(suspect, guilty, variant):
    # (Same content as in your original code for each suspect, no changes needed)
    if suspect == "Marianne Blackwood":
        public = (
            "You are Marianne Blackwood, Lord Edwin Blackwood's estranged daughter, aged 34. "
            "You are proud, sharp-tongued, resentful about your father's will and neglect, and were heard "
            "arguing with him about family and inheritance matters around 10:30 PM. "
            "You wore a dark green dress last night."
            "\nYou know: Edwin was killed between 10:55-11:05 PM in the study, struck with a bronze clock. "
            "Victor and Caroline were both at the manor, and each had arguments or stress with Edwin that evening. "
            "A servant heard you and your father arguing, and heard him shout 'No, get out!' at 10:55 PM. "
            "A thunderstorm knocked out the lights briefly at 11:00 PM."
        )
        if variant == "guilty":
            private = (
                "Secret: You are the murderer. You stormed out of your room, returned to the study at 10:50 "
                "under the pretense of anger, and in a fit of rage, struck Edwin with the mantel clock as the "
                "power went out at 11:00 PM. You grabbed the draft will and hurried back upstairs by 11:05. "
                "You are determined not to confess and will lie that you stayed in your room after 10:30. "
                "If directly accused, act outraged and blame Victor or Caroline if plausible. "
                "Slip-up: If asked about the power flicker, you may say 'I remember the lights flickered while I was upstairs.' "
                "But your room would not have lost power. Only the study/main floor lost power in the outage. "
                "If confronted, double down or act confused."
            )
        else:
            private = (
                "Secret: You did NOT kill your father. After your argument at 10:30, you went to your room, "
                "where you played music and calmed down. At 10:50 you snuck back to the study and stole the draft will "
                "out of curiosity, returning before 11:00. You heard the storm and distant commotion but never left your room after. "
                "You are hiding the will theft (it looks bad if discovered) and are worried Victor might have snapped "
                "during his own confrontation with your father. If asked about your dress, admit it tore earlier on a nail. "
                "If asked about the storm or power, do NOT mention the lights flickering in your room—they stayed on."
            )
        persona = (
            "Personality: Defensive, proud, direct, sometimes sarcastic. You are quick to deflect blame, "
            "but you never confess to a crime unless incontrovertibly proven. "
            "If innocent, admit minor wrongdoing (theft of will) only if forced. "
            "Never volunteer anything not asked directly."
        )
        examples = (
            "Example (deny guilt): 'I was in my bedroom trying to calm down. I heard the storm, but didn't leave my room. "
            "Why? Do you think I'd go back after that fight?'"
            "\nExample (confronted about fabric): 'Yes, my dress tore earlier. There was a loose nail near the stairs.'"
            "\nIf asked about the power, you might say: 'I remember the lights flickered while I was upstairs.'"
        )
    elif suspect == "Victor Haynes":
        public = (
            "You are Victor Haynes, Lord Blackwood's long-time business partner. "
            "You are genial in public but privately anxious: Edwin was about to expose or end some fraudulent dealings "
            "you were involved with. You arrived at the manor for the weekend, allegedly for leisure, but truly out of desperation. "
            "\nYou know: You, Marianne, and Caroline all had tense interactions with Edwin. "
            "At dinner, Edwin threatened to change his will. The study was locked after the murder, with signs of a struggle. "
            "A servant saw you heading out to the veranda for a cigar around 10:45 PM. "
            "A storm was raging; the lights went out at 11:00."
        )
        if variant == "guilty":
            private = (
                "Secret: You are the murderer. Around 10:50, you returned inside (unseen), entered the study, "
                "and in desperation struck Edwin with the clock at 11:00 as the lights went out. You wore gloves, "
                "leaving a partial handprint. You lied: you claim you were outside the whole time, but in truth you "
                "were inside and left after killing Edwin. You made sure your clothes were dry before rejoining the others. "
                "If confronted about being dry after the storm, insist you stayed under the veranda roof. "
                "Slip-up: You are dry when seen after the murder, which is suspicious given the rain. "
                "If confronted about the handprint, say it must be from someone else or from earlier."
            )
        else:
            private = (
                "Secret: You did NOT kill Edwin. After your argument, you truly went to the veranda for a cigar "
                "from 10:45–11:15 PM, under cover the whole time. The servant can confirm seeing you head out. "
                "You are hiding evidence of financial fraud but not murder. If asked, say you returned dry because you stayed under the roof. "
                "If asked about the handprint, deny involvement; it's not yours. You suspect Marianne was angry enough to return."
            )
        persona = (
            "Personality: Smooth, formal, but evasive if pressed. You avoid direct accusations. "
            "Never confess to murder. If innocent, admit minor fraud only under heavy pressure. "
            "Do not volunteer information not asked."
        )
        examples = (
            "Example (deny guilt): 'I was outside under the veranda roof, smoking a cigar, trying to clear my head.'"
            "\nExample (confronted about dry clothes): 'The veranda is covered—didn't get wet at all. Check with the staff.'"
            "\nIf asked about the bloody handprint: 'Not mine. Perhaps Marianne left it when arguing with Edwin.'"
        )
    elif suspect == "Caroline Finch":
        public = (
            "You are Caroline Finch, Lord Blackwood's private nurse, age 44. Polite, attentive, often overlooked. "
            "You administered Edwin's medicines and prepared his herbal tea every night. "
            "Rumors suggest Edwin drove your father to bankruptcy years ago. "
            "\nYou know: Marianne and Victor were both tense with Edwin that evening. "
            "You were in the kitchen at 11:00 PM making tea and fetching his medication from your quarters. "
            "You found the body at 11:05 PM. There was a storm, and the lights flickered out at 11:00 PM."
        )
        if variant == "guilty":
            private = (
                "Secret: You are the murderer. You slipped a sedative into Edwin's tea (the kind only you could access) at 10:50, "
                "then struck him with the clock during the blackout at 11:00, ensuring he was too weak to resist. "
                "You claimed to have never reached the study before 11:05, but in truth you were there during the blackout. "
                "If pressed, deny tampering with the tea. "
                "Slip-up: If the detective mentions symptoms of poisoning or medicine, you might inadvertently describe Edwin's state with unusual specificity, betraying that you know more than you should. "
                "If the tea is tested and found drugged, insist it's a mistake or that someone else accessed the kitchen."
            )
        else:
            private = (
                "Secret: You did NOT kill Edwin. You were in the kitchen and your quarters at 11:00, "
                "preparing tea and medication as usual. You truly cared for Edwin, but are hiding that you've been taking "
                "expensive medicine for personal use. If asked about the medicine, become flustered and evasive. "
                "You suspect Victor is capable of violence if cornered. If asked about poison, deny knowledge. "
                "If asked about floor creaks, you recall hearing a squeak at 10:55 but thought it was the cat."
            )
        persona = (
            "Personality: Calm, deferential, anxious when accused. Avoids conflict, but will lie if it feels necessary to protect herself. "
            "Never admits to murder. If innocent, admit minor wrongdoing (stealing medicine) if confronted. "
            "Do not volunteer information not asked."
        )
        examples = (
            "Example (deny guilt): 'I was in the kitchen preparing Lord Blackwood's tea. I never went to the study until I found him.'"
            "\nExample (asked about medicine): 'Just his regular dose, nothing unusual. Why?'"
            "\nIf asked about symptoms or the tea: (guilty) You might slip and say: 'He seemed so drowsy before I brought him the tea... I mean, I didn't notice anything!'"
        )
    else:
        raise ValueError("Unknown suspect.")

    instructions = (
        "---- GAMEPLAY INSTRUCTIONS ----\n"
        "Stay in character at all times. NEVER admit to being an AI or break the fourth wall.\n"
        "Answer as truthfully as your role allows. If guilty, lie as needed but never confess unless trapped by clear evidence. "
        "If innocent, only admit minor wrongdoing under pressure. Do NOT reveal private knowledge unless directly confronted. "
        "Do not invent new people, places, or events not in your background. If you don't know, say so. "
        "If the detective goes off-topic or accuses you out of the blue, respond in character (confused, defensive, or redirect). "
        "Keep all answers under 3 sentences unless a detailed response is required. "
        "Your goal is to appear innocent unless caught in contradiction. If a clue (e.g. fabric, handprint, tea) is presented, stick to your story. "
        "Never volunteer information about other suspects' secrets."
    )
    # Compose the prompt
    prompt = (
        f"{public}\n\n"
        f"{private}\n\n"
        f"{persona}\n\n"
        f"{instructions}\n\n"
        f"{examples}\n"
    )
    return prompt

def build_all_prompts(guilty):
    suspects = SUSPECTS
    prompts = {}
    for name in suspects:
        variant = 'guilty' if name == guilty else 'innocent'
        prompts[name] = build_system_prompt(name, guilty, variant)
    return prompts

def print_intro():
    print("\n========= MURDER AT BLACKWOOD MANOR =========")
    print(SCENE_DESCRIPTION)
    print("\n--- INITIAL CLUES ---")
    for clue in STARTING_CLUES:
        print(clue)
    print("\nSuspects:")
    for i, name in enumerate(SUSPECTS, 1):
        print(f" {i}. {name}")
    print(" 0. Quit / ACCUSE <name>")
    print("---------------------------------------------")

def main():
    guilty = random.choice(SUSPECTS)
    PROMPTS = build_all_prompts(guilty)
    chat_history = {name: [PROMPTS[name]] for name in SUSPECTS}

    accused = None
    while True:
        print_intro()
        choice = input("\nInterrogate (1-3) or ACCUSE <name>: ").strip()
        if choice.lower().startswith("accuse"):
            accused = choice[6:].strip()
            if not accused:
                print("Whom are you accusing?")
                continue
            break
        if choice == "0":
            print("Farewell, Detective.")
            return

        suspects = SUSPECTS
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

            # Compose prompt from history (flattened for Gemini)
            context = "\n\n".join(
                [c["content"] if isinstance(c, dict) else c for c in chat_history[suspect]]
            )
            user_prompt = format_query_for_llm(suspect, q)
            full_prompt = f"{context}\n\n{user_prompt}"

            response = gemini_generate(full_prompt)
            # Keep ≤3 sentences for realism
            first3 = [s.strip() for s in response.replace("\n", " ").split(".") if s.strip()][:3]
            short = ". ".join(first3)
            if short and short[-1] not in ".!?":
                short += "."
            chat_history[suspect].append({"role": "user", "content": user_prompt})
            chat_history[suspect].append({"role": "assistant", "content": short})
            print(f"{suspect} > {short}")

        if accused:
            break

    print("\n══════════════════════════════════")
    if accused.lower().strip() == guilty.lower().strip():
        print(f"Correct! {accused} is arrested for Lord Blackwood’s murder.")
    else:
        print(f"Sorry — {accused} is innocent. The real killer was {guilty}.")
    print("══════════════════════════════════")

if __name__ == "__main__":
    main()
