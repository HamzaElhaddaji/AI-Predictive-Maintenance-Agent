"""
Agent de maintenance predictive (Phase 6).

Deux agents avec la meme interface `run(question) -> str` :
 - LLMAgent    : un LLM decide quels tools appeler (tool calling, API compatible OpenAI :
                 Groq, Gemini, Ollama, OpenRouter...)
 - ManualAgent : regles simples, sans LLM (gratuit, sert de repli)

Configuration LLM par variables d'environnement :
    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
"""

import json
import os
import re

from src.agent.tool_registry import TOOL_SCHEMAS, execute_tool

SYSTEM_PROMPT = """Tu es un agent de maintenance predictive pour une flotte de moteurs.
Tu disposes d'outils qui donnent la duree de vie restante (RUL, en cycles) et le niveau
de risque (critique, eleve, modere, faible) des machines.

Regles :
- Utilise les outils pour obtenir les chiffres : n'invente jamais une valeur.
- Choisis seulement les outils necessaires a la question.
- Reponds en francais, de facon courte et claire, en citant les valeurs obtenues.
- Si un outil renvoie une erreur, explique-la simplement a l'utilisateur.
"""


class LLMAgent:
    def __init__(self, client=None, model=None, max_steps=6, verbose=True):
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=os.environ["LLM_BASE_URL"],
                api_key=os.environ["LLM_API_KEY"],
            )
        self.client = client
        self.model = model or os.environ["LLM_MODEL"]
        self.max_steps = max_steps
        self.verbose = verbose

    def run(self, question):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        for _ in range(self.max_steps):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0,
            )
            msg = resp.choices[0].message

            if not msg.tool_calls:  # reponse finale
                return msg.content

            messages.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                    result = execute_tool(call.function.name, args)
                except json.JSONDecodeError:
                    result = {"error": "Arguments du tool invalides (JSON)."}
                if self.verbose:
                    print(f"[tool] {call.function.name}({call.function.arguments}) -> {result}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
        return "Je n'ai pas reussi a conclure en un nombre raisonnable d'etapes."


class ManualAgent:
    """Agent a regles : memes tools, mais le choix est code a la main."""

    def __init__(self, verbose=True):
        self.verbose = verbose

    def _call(self, name, args=None):
        result = execute_tool(name, args)
        if self.verbose:
            print(f"[tool] {name}({args or {}}) -> {result}")
        return result

    def run(self, question):
        q = question.lower()
        m = re.search(r"(?:machine|moteur|unit|engine)\s*#?\s*(\d+)", q)
        if m:
            r = self._call("get_machine_risk", {"unit_id": int(m.group(1))})
            if "error" in r:
                return f"Erreur : {r['error']}"
            return (
                f"Machine {r['unit_id']} : RUL predit ~ {r['rul_predit']} cycles, "
                f"niveau de risque « {r['niveau_risque']} »."
            )
        if any(k in q for k in ["critique", "risque", "danger", "urgent", "priorit", "liste"]):
            r = self._call("list_high_risk_machines", {"max_level": "eleve"})
            if "error" in r:
                return f"Erreur : {r['error']}"
            top = ", ".join(
                f"{x['unit_id']} ({x['rul_predit']} cycles)" for x in r["machines"][:10]
            )
            return f"{r['count']} machine(s) a risque eleve ou critique. Les plus urgentes : {top}."
        r = self._call("get_fleet_summary")
        if "error" in r:
            return f"Erreur : {r['error']}"
        return (
            f"{r['n_machines']} machines, RUL moyen {r['rul_moyen']} cycles. "
            f"Repartition : {r['repartition']}. "
            f"La plus critique : machine {r['machine_la_plus_critique']['unit_id']}."
        )


def build_agent(verbose=True):
    """LLMAgent si les variables d'environnement LLM sont definies, sinon ManualAgent."""
    if all(k in os.environ for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")):
        return LLMAgent(verbose=verbose)
    return ManualAgent(verbose=verbose)


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "Analyse la machine 5"
    print(build_agent().run(question))
