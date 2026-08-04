import tiktoken

ENCODING = None


def get_encoding():
    global ENCODING
    if ENCODING is None:
        ENCODING = tiktoken.get_encoding("cl100k_base")
    return ENCODING

def count_tokens(text: str) -> int:
    return len(get_encoding().encode(text))


def truncate_text(text: str, max_tokens: int) -> str:
    encoding = get_encoding()
    tokens = encoding.encode(text)
    return encoding.decode(tokens[:max_tokens])
