"""Competitor strategy definition for starter-kit agents.

Customized for Team Sadeeq in the NVIDIA Developer Live Agent Gauntlet.
"""

from __future__ import annotations

from typing import Any
from base_strategy import BaseStrategy, ChallengeContext


class MyStrategy(BaseStrategy):
    """Competitive custom strategy for Team Sadeeq."""

    # Team identity (used in Organizer / Dashboard)
    agent_id = "sadeeq-gauntlet"
    agent_name = "Team Sadeeq"

    # Preferred models for text tasks (ranked by speed, reasoning accuracy, and token efficiency)
    PREFERRED_TEXT_MODELS = [
        "gemini-3.8-flash",
        "qwen3.7-flash",
        "gemini-3.6-flash",
        "qwen3.7-plus",
        "nemotron-3.5-lightning",
        "nemotron-3-super",
        "qwen3.6-35b",
    ]

    # Preferred models for image tasks
    PREFERRED_IMAGE_MODELS = [
        "gemini-3.1-flash-image",
        "gemini-3-pro-image",
        "gemini-3.1-flash-lite-image",
    ]

    # Text challenge configuration
    text_system_prompt = (
        "You are an expert competitive challenge solver participating in the Agent Gauntlet.\n"
        "Your first line must ALWAYS be: ANSWER: <final answer>\n"
        "Follow every constraint and rule in the challenge description exactly.\n"
        "Do NOT include markdown formatting, backticks, or code blocks around the ANSWER.\n"
        "Do NOT include <think> tags.\n"
        "Work briskly and concisely. Do not over-deliberate or second-guess once constraints are satisfied.\n"
        "After the ANSWER line, you may optionally include at most 2 brief lines of reasoning.\n"
        "Never output 'unknown' or empty answers."
    )
    text_strategy_notes = "Fast, accurate greedy decoding with strict answer formatting."
    text_temperature = 0.0
    text_max_tokens = 2048

    # Image challenge configuration
    image_strategy_notes = (
        "Prioritize image_edit when source image is present; otherwise use image_generate. "
        "Adhere strictly to style, subject, and composition rules with standard resolution."
    )

    def rank_models(
        self,
        ctx: ChallengeContext,
        available_models: list[str],
    ) -> list[str]:
        """Rank models dynamically based on modality and performance profiles."""
        if self.is_image_challenge(ctx):
            image_candidates = self.prefer_image_models(available_models)
            ranked: list[str] = []
            for preferred in self.PREFERRED_IMAGE_MODELS:
                if preferred in image_candidates and preferred not in ranked:
                    ranked.append(preferred)
            for m in image_candidates:
                if m not in ranked:
                    ranked.append(m)
            for m in available_models:
                if m not in ranked:
                    ranked.append(m)
            return ranked

        # Text challenge ranking
        ranked: list[str] = []
        for preferred in self.PREFERRED_TEXT_MODELS:
            if preferred in available_models and preferred not in ranked:
                ranked.append(preferred)
        for m in available_models:
            if m not in ranked:
                ranked.append(m)
        return ranked

    def pick_model(
        self,
        stage: str,
        ranked_models: list[str],
        ctx: ChallengeContext,
    ) -> str:
        """Choose the best model for the current stage and challenge context."""
        if not ranked_models:
            return ""
        return ranked_models[0]

    def build_system_prompt(self, ctx: ChallengeContext) -> str:
        """Build the system prompt for the challenge."""
        return str(self.text_system_prompt).strip()

    def build_solver_prompt(self, ctx: ChallengeContext) -> str:
        """Build the user solver prompt with structured clues and constraints."""
        clue_lines = "\n".join(f"- {clue}" for clue in ctx.clues if str(clue).strip())
        if not clue_lines:
            clue_lines = "- (No clues provided.)"

        challenge_type = ctx.challenge_type.strip() or "text"
        description = ctx.description.strip() or "Solve the problem."
        rules = ctx.rules.strip() or "No additional constraints."

        return (
            "Solve the challenge below accurately and concisely.\n\n"
            f"Challenge Type: {challenge_type}\n"
            f"Description: {description}\n\n"
            f"Rules and Constraints:\n{rules}\n\n"
            f"Clues:\n{clue_lines}\n\n"
            "Requirements:\n"
            "1. Line 1 must be: ANSWER: <your exact final answer>\n"
            "2. Follow all formatting specifications from the Rules (casing, delimiter, ordering).\n"
            "3. If reasoning is included, keep it under 2 brief lines below ANSWER."
        )

    def get_llm_params(self, ctx: ChallengeContext) -> dict[str, Any]:
        """Return parameters optimized for determinism and efficiency."""
        return {
            "temperature": float(self.text_temperature),
            "max_tokens": int(self.text_max_tokens),
        }

    def should_submit_early(self, answer: str, ctx: ChallengeContext) -> bool:
        """Submit early if a valid answer is extracted and time is running low."""
        cleaned = str(answer or "").strip()
        if not cleaned:
            return False
        # If less than 15s remaining, submit immediately
        if ctx.time_remaining_s and ctx.time_remaining_s < 15.0:
            return True
        return False

    def on_time_warning(
        self,
        remaining_s: float,
        current_answer: str,
        ctx: ChallengeContext,
    ) -> str | None:
        """Fallback safeguard when nearing timeout."""
        if current_answer and current_answer.strip():
            return current_answer.strip()
        return None

    def plan_image_tool(
        self,
        ctx: ChallengeContext,
        available_tools: list[str],
    ) -> str:
        """Select appropriate image tool."""
        if ctx.image_url and "image_edit" in available_tools:
            return "image_edit"
        if "image_generate" in available_tools:
            return "image_generate"
        if ctx.image_url and "image_analyze" in available_tools:
            return "image_analyze"
        return available_tools[0] if available_tools else ""

    def build_image_prompt(self, ctx: ChallengeContext) -> str:
        """Construct prompt for image generation/editing."""
        headline = ctx.challenge_text.strip() or ctx.description.strip()
        details = ctx.rules.strip()
        if details and headline and headline not in details:
            base = f"{details}\n\nContext: {headline}"
        else:
            base = details or headline or "Create a high quality image matching requirements."
        return f"{base}\n\nStandard resolution, accurate to description, no unwanted artifacts."
