from __future__ import annotations

from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape


_ENV = Environment(
    loader=PackageLoader("multi_agent_sync.prompts", "templates"),
    autoescape=select_autoescape(default=False),
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)


def render_prompt(template_name: str, **context: Any) -> str:
    template = _ENV.get_template(template_name)
    return template.render(**context).strip()
