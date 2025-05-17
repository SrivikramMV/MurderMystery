from dataclasses import dataclass
from typing import List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", trust_remote_code=True
    )
    return tokenizer, model


@dataclass
class Suspect:
    name: str
    role: str
    personality: str
    motive: str
    alibi: str
    secret: str
    is_murderer: bool


SUSPECTS: List[Suspect] = [
    Suspect(
        name="Alice Carpenter",
        role="Victor's business partner",
        personality="formal and precise",
        motive="financial gain from Victor's death",
        alibi="was at a charity gala during the murder",
        secret="has been forging company documents",
        is_murderer=False,
    ),
    Suspect(
        name="Brian Doyle",
        role="long-time friend",
        personality="nervous and talkative",
        motive="jealous of Victor's success",
        alibi="claims he was at home watching TV",
        secret="owes huge gambling debts",
        is_murderer=True,
    ),
    Suspect(
        name="Cynthia Everly",
        role="mysterious neighbor",
        personality="calm and observant",
        motive="grudge over a property dispute",
        alibi="seen walking her dog that evening",
        secret="hired a private investigator to spy on Victor",
        is_murderer=False,
    ),
]

VICTIM = "Victor Lang"


def system_prompt(suspect: Suspect) -> str:
    return (
        f"You are {suspect.name}, {suspect.role}. "
        f"Your personality is {suspect.personality}. "
        f"Your motive: {suspect.motive}. "
        f"Public alibi: {suspect.alibi}. "
        f"Hidden secret: {suspect.secret}. "
        "Never reveal your secret or guilt unless confronted with direct evidence. "
        "Answer in natural language, 1-3 sentences unless the detective asks for detail."
    )


def generate_response(history: List[Dict[str, str]], tokenizer, model, max_new_tokens: int = 128) -> str:
    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    reply = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return reply.strip()


def accusation(suspects: List[Suspect], name: str) -> bool:
    for s in suspects:
        if name.lower() in s.name.lower():
            return s.is_murderer
    return False


def main():
    tokenizer, model = load_model()
    histories: Dict[str, List[Dict[str, str]]] = {
        s.name: [{"role": "system", "content": system_prompt(s)}] for s in SUSPECTS
    }

    print(f"{VICTIM} was found murdered.\n")
    print("Suspects:")
    for idx, s in enumerate(SUSPECTS, 1):
        print(f"{idx}. {s.name}")
    print()

    while True:
        choice = input("Talk to which suspect? (0 = quit) ").strip()
        if choice == "0":
            print("Good-bye detective.")
            return
        if not choice.isdigit() or not (1 <= int(choice) <= len(SUSPECTS)):
            continue
        suspect = SUSPECTS[int(choice) - 1]
        history = histories[suspect.name]

        while True:
            user_input = input(f"{suspect.name} > ").strip()
            if user_input.lower() == "menu":
                break
            if user_input.lower() == "accuse":
                accused = input("Who do you accuse? ").strip()
                if accusation(SUSPECTS, accused):
                    print("Correct! Justice is served.")
                else:
                    murderer = next(s.name for s in SUSPECTS if s.is_murderer)
                    print(f"Wrong! The murderer was {murderer}.")
                return
            if user_input.lower() == "quit":
                print("Good-bye detective.")
                return
            history.append({"role": "user", "content": user_input})
            try:
                reply = generate_response(history, tokenizer, model)
            except Exception as e:
                reply = f"[Model error: {e}]"
            history.append({"role": "assistant", "content": reply})
            print(reply)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGood-bye detective.")

