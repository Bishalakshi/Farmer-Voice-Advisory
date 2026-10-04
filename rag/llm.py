"""Single shared 4-bit LLM instance (llama.cpp GGUF)."""
import config
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        _llm = Llama.from_pretrained(repo_id=config.LLM_REPO, filename=config.LLM_FILE,
                                     n_ctx=config.LLM_CTX, n_gpu_layers=config.LLM_GPU_LAYERS, verbose=False)
    return _llm


def chat(messages, temperature=0.1, max_tokens=300, json_mode=False):
    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    r = get_llm().create_chat_completion(messages=messages, temperature=temperature, max_tokens=max_tokens, **kw)
    return r["choices"][0]["message"]["content"].strip()
