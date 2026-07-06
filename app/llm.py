import sys
import time
import logging
from typing import List, Dict, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging to console (and optionally to file later)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler = logging.FileHandler("logs/app.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)
logger = logging.getLogger(__name__)

class LLMUnavailable(Exception):
    """Custom exception for any model loading or runtime error."""
    pass

_model = None
_tokenizer = None
_device = None

def _load_model():
    """Load the Qwen2.5-0.5B model and tokenizer once."""
    global _model, _tokenizer, _device
    try:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model Qwen/Qwen2.5-0.5B-Instruct on {_device}...")
        # For some model versions you might need trust_remote_code=True.
        # The tokenizer is safe; the model may require it if not natively supported.
        _tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            trust_remote_code=True        # safe for Qwen
        )
        _model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            trust_remote_code=True
        )
        _model.to(_device)
        _model.eval()                     # disable dropout, etc.
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise LLMUnavailable(f"Model loading failed: {e}") from e

# Load immediately on first import
_load_model()

def generate(messages: List[Dict[str, str]], max_tokens: int = 256) -> str:
    """
    Generate a response from the loaded Qwen model.

    Args:
        messages: List of chat messages (role/content dicts).
        max_tokens: Maximum number of tokens to generate.

    Returns:
        The generated text (assistant response).

    Raises:
        LLMUnavailable: If any error occurs during generation.
    """
    if _model is None or _tokenizer is None:
        raise LLMUnavailable("Model not loaded. This should not happen.")

    try:
        start_time = time.time()

        # Apply the chat template -> string
        prompt = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True   # adds assistant header
        )

        # Tokenize the prompt
        inputs = _tokenizer(prompt, return_tensors="pt").to(_device)
        input_token_count = inputs.input_ids.shape[1]

        # Generate with temperature=0 (deterministic)
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.0,          # no randomness
                do_sample=False,          # greedy decoding
                pad_token_id=_tokenizer.eos_token_id
            )

        # Decode the full output (includes prompt)
        full_output = _tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant part
        # The chat template produces a final "assistant" marker; we take everything after the prompt.
        # Simpler: decode only the new tokens
        response = _tokenizer.decode(
            outputs[0][input_token_count:],
            skip_special_tokens=True
        ).strip()

        output_token_count = outputs.shape[1] - input_token_count
        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(
            f"LLM call: input_tokens={input_token_count}, "
            f"output_tokens={output_token_count}, "
            f"latency_ms={elapsed_ms:.2f}"
        )

        return response

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise LLMUnavailable(f"Generation failed: {e}") from e
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python app/llm.py \"Your message here\"")
        sys.exit(1)

    user_message = sys.argv[1]
    messages = [{"role": "user", "content": user_message}]

    try:
        reply = generate(messages)
        print(reply)
    except LLMUnavailable as e:
        logger.error(f"LLM call failed: {e}")
        sys.exit(1)
    