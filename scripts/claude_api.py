import os
import json
import logging
from typing import Any

import anthropic

from pathlib import Path

logger = logging.getLogger(__name__)

MODEL = os.getenv(
    "ANTHROPIC_MODEL",
    "claude-sonnet-4-20250514",
)

api_key = (
    os.getenv("ANTHROPIC_AUTH_TOKEN", "")
    or os.getenv("ANTHROPIC_API_KEY", "")
    or os.getenv("ANTHROPIC_TOKEN", "")
)
base_url = os.getenv(
    "ANTHROPIC_BASE_URL",
    "https://api.anthropic.com",
)

if not api_key:
    raise RuntimeError(
        "ANTHROPIC_AUTH_TOKEN is not set. "
        "Configure it in GitHub Actions secrets."
    )

# Build client kwargs conditionally to avoid
# passing empty values to newer SDK versions.
client_kwargs: dict[str, Any] = {}

if api_key:
    client_kwargs["api_key"] = api_key

if base_url:
    client_kwargs["base_url"] = base_url.rstrip("/")

client = anthropic.Anthropic(**client_kwargs)


def load_skill(skill_name: str) -> str:
    project_root = Path(__file__).resolve().parent.parent

    skill_file = (
        project_root
        / "skills"
        / skill_name
        / "skills.md"
    )

    if not skill_file.exists():
        raise FileNotFoundError(
            f"Skill not found: {skill_file}"
        )

    return skill_file.read_text(
        encoding="utf-8"
    )


def review_code(
    diff: str,
    repository: str,
    pr_number: int,
    selected_skills: list[str] | None = None,
    review_mode: str = "PR",
    repository_context: str = "",
    **kwargs,
):
    # Load ONLY the skills selected by the user
    skills_content = []

    # Support both 'selected_skills' and 'skills' param names
    if selected_skills is None:
        selected_skills = kwargs.get("skills") or []

    selected_skills = selected_skills or []

    for skill_name in selected_skills:
        skill_name = skill_name.strip()

        if not skill_name:
            continue

        skill = load_skill(skill_name)

        skills_content.append(
            f"""
==============================
SKILL: {skill_name}
==============================
{skill}
"""
        )

    # If no skill was selected, use code-review as default
    if not skills_content:
        skills_content.append(
            f"""
==============================
SKILL: code-review
==============================
{load_skill("code-review")}
"""
        )

    combined_skills = "\n".join(skills_content)

    prompt = f"""
You must perform this task using the following selected skills.

{combined_skills}

==============================
REVIEW CONTEXT
==============================
Repository:
{repository}

Review Mode:
{review_mode}

Pull Request:
#{pr_number}

Repository Context:
{repository_context}

==============================
CODE / DIFF
==============================
{diff}

==============================
INSTRUCTIONS
==============================
Follow all selected skill instructions exactly.

Return ONLY the JSON format specified
by the applicable skill instructions.
"""

    print(f"Calling Claude model: {MODEL}")

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    text = response.content[0].text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    try:
        review = json.loads(text)

    except json.JSONDecodeError as exc:
        print("Claude returned invalid JSON:")
        print(text)

        raise RuntimeError(
            "Claude response was not valid JSON."
        ) from exc

    return review
