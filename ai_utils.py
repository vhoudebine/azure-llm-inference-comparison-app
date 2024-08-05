import tiktoken
import re


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def is_valid_url(string: str, model_type: str) -> bool:

    if model_type == "AOAI":
        return string.startswith("http://") or string.startswith("https://") and string.endswith("openai.azure.com") or string.endswith("openai.azure.com/")
    elif model_type == "Azure MaaS":
        return string.startswith("http://") or string.startswith("https://") and string.endswith(".inference.ai.azure.com") or string.endswith(".inference.ai.azure.com/")
    elif model_type == "Azure MaaP":
        return string.startswith("http://") or string.startswith("https://") and string.endswith(".inference.ml.azure.com") or string.endswith("inference.ml.azure.com/")
