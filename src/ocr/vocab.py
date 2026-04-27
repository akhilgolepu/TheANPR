# Standard Indian License Plate characters (A-Z, 0-9)
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# CTC requires index 0 to be the 'blank' token. We shift characters by 1.
char2int = {c: i + 1 for i, c in enumerate(CHARS)}
int2char = {i + 1: c for i, c in enumerate(CHARS)}

NUM_CLASSES = len(CHARS) + 1  # 36 chars + 1 CTC blank = 37

def encode_label(text: str) -> list:
    """Converts 'MH02' into [23, 18, 1, 3]"""
    return [char2int[c] for c in text.upper() if c in char2int]

def decode_label(int_list: list) -> str:
    """Converts [23, 18, 1, 3] back into 'MH02'"""
    return "".join([int2char[i] for i in int_list if i != 0])