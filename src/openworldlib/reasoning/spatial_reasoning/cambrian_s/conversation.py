from __future__ import annotations

from typing import Any, Iterable

from .constants import DEFAULT_IMAGE_TOKEN, DEFAULT_SYSTEM_PROMPT


def _iter_content_items(content: Any) -> Iterable[dict[str, Any]]:
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                yield item


def render_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    image_prefix_parts: list[str] = []
    text_parts: list[str] = []
    for item in _iter_content_items(content):
        item_type = item.get("type")
        if item_type in {"image", "video"}:
            image_prefix_parts.append(DEFAULT_IMAGE_TOKEN)
        elif item_type == "text":
            text = str(item.get("text", ""))
            if text:
                text_parts.append(text)

    return "".join(image_prefix_parts) + "".join(text_parts)


def extract_media_inputs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    media_inputs: list[dict[str, Any]] = []
    for message in messages:
        for item in _iter_content_items(message.get("content")):
            if item.get("type") in {"image", "video"}:
                media_inputs.append(item)
    return media_inputs


def normalize_messages(
    messages: list[dict[str, Any]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> tuple[str, list[dict[str, str]]]:
    normalized: list[dict[str, str]] = []
    resolved_system_prompt = system_prompt

    for message in messages:
        role = str(message.get("role", "user"))

        text = render_message_content(message.get("content", ""))
        if role == "system":
            resolved_system_prompt = text or system_prompt
        elif role == "assistant":
            normalized.append({"role": "assistant", "text": text})
        else:
            normalized.append({"role": "user", "text": text})

    return resolved_system_prompt, normalized


def build_qwen_chat_prompt(
    messages: list[dict[str, Any]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    resolved_system_prompt, normalized_messages = normalize_messages(
        messages,
        system_prompt=system_prompt,
    )
    prompt = f"<|im_start|>system\n{resolved_system_prompt}<|im_end|>\n"
    role_prefixes = {
        "user": "<|im_start|>user",
        "assistant": "<|im_start|>assistant",
    }
    for message in normalized_messages:
        role_prefix = role_prefixes[message["role"]]
        if message["text"]:
            prompt += f"{role_prefix}\n{message['text']}<|im_end|>\n"
        else:
            prompt += f"{role_prefix}\n"
    prompt += "<|im_start|>assistant\n"
    return prompt
