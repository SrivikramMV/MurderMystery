import os
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "2"

from dataclasses import dataclass
from typing import List, Dict

import torch
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MODEL_REPO = "Qwen/Qwen3-0.6B-GGUF"
MODEL_FILE = "Qwen3-0.6B-Q4_K_M.gguf"

llm: Llama


def load_model() -> Llama:
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return Llama(
        model_path=model_path,
        chat_format="qwen",
        n_ctx=1024,
        n_gpu_layers=-1 if torch.cuda.is_available() else 0,
    )

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

def generate_response(history: List[Dict[str, str]]) -> str:
    reply = llm.create_chat_completion(history)["choices"][0]["message"]["content"]
    return reply.strip()


def accusation(suspects: List[Suspect], name: str) -> bool:
    for s in suspects:
        if name.lower() in s.name.lower():
            return s.is_murderer
    return False


def main():
    global llm
    llm = load_model()

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
                reply = generate_response(history)
            except Exception as e:
                reply = f"[Model error: {e}]"
            history.append({"role": "assistant", "content": reply})
            print(reply)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGood-bye detective.")

