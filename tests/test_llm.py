from multi_agent_sync import llm as llm_module


def test_get_llm_passes_max_tokens_to_openai_compatible_provider(monkeypatch):
    captured_kwargs = {}

    def fake_chat_openai(**kwargs):
        captured_kwargs.update(kwargs)
        return "openai-llm"

    monkeypatch.setattr(llm_module, "ChatOpenAI", fake_chat_openai)

    llm = llm_module.get_llm(model="test-model", openai=True, max_tokens=8192)

    assert llm == "openai-llm"
    assert captured_kwargs["model"] == "test-model"
    assert captured_kwargs["max_completion_tokens"] == 8192


def test_get_llm_passes_max_tokens_to_ollama_provider(monkeypatch):
    captured_kwargs = {}

    def fake_chat_ollama(**kwargs):
        captured_kwargs.update(kwargs)
        return "ollama-llm"

    monkeypatch.setattr(llm_module, "ChatOllama", fake_chat_ollama)

    llm = llm_module.get_llm(model="qwen3:4b", openai=False, max_tokens=8192)

    assert llm == "ollama-llm"
    assert captured_kwargs["model"] == "qwen3:4b"
    assert captured_kwargs["num_predict"] == 8192
