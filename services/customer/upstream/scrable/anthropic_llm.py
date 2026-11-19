"""Claude backend using the official Anthropic SDK (kept separate from the OpenAI-compatible backend)."""

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept server-side refusal fallbacks with fallbacks="default".
FALLBACK_MODELS = {
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
    "claude-fable-5-2",
    "claude-fable-5-1",
}


class AnthropicLLM:
    """Messages API backend. Reads ANTHROPIC_API_KEY (or an `ant auth login` profile)."""

    def __init__(self, model, min_output_tokens=16000, effort=None, client=None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=3)
        self.client = client
        self.model_name = model
        self.min_output_tokens = min_output_tokens
        self.effort = effort
        self.fallbacks = model in FALLBACK_MODELS
        self.identity = {
            "backend": "anthropic",
            "model": model,
            "min_output_tokens": min_output_tokens,
            "effort": effort,
            "fallbacks": "default" if self.fallbacks else None,
            "request_version": 1,
        }

    def complete(self, messages, max_tokens):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        conversation = [
            {"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"
        ]
        # Adaptive thinking spends output tokens, so small role budgets act only as a floor.
        kwargs = {
            "model": self.model_name,
            "max_tokens": max(max_tokens, self.min_output_tokens),
            "messages": conversation,
        }
        if system:
            kwargs["system"] = system
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.fallbacks:
            response = self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
        else:
            response = self.client.messages.create(**kwargs)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise RuntimeError(f"Claude refused the request: {getattr(details, 'explanation', None)}")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("Claude reached max_tokens; increase api_min_output_tokens")
        return "".join(block.text for block in response.content if block.type == "text").strip()
