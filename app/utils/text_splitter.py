from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

def split_text(text, max_tokens=5000):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ''
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence))
        if current_tokens + sentence_tokens > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
            current_tokens = sentence_tokens
        else:
            current_chunk += sentence + '. '
            current_tokens += sentence_tokens

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
