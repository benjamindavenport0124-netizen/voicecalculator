import threading
import tkinter as tk
from requests import get
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import wavio
import time
import math
import re
from sympy import I as symI
import sympy as sp
from sympy import N, symbols, nsolve, solve, exp, Eq, solve_poly_system, sympify, sin, cos, tan, asin, acos, atan, log, pi, E, sqrt, I, lambdify, simplify, Abs
from sympy.core.sympify import SympifyError
from sympy.solvers.solveset import linear_eq_to_matrix
from scipy.optimize import root
from mpmath import mp

mp.dps = 50

# Set to True to enable detailed debug printouts while developing or debugging
DEBUG = False

use_radians = True

substitution = False
# --- Load Whisper model once ---
model = whisper.load_model("medium")


recording = False
frames = []
fs = 16000

#------------- Recording functions -------------
def start_recording():
    global recording, frames
    if recording:
        return  # already recording, ignore
    recording = True
    frames = []
    start_button.config(state="disabled")
    stop_button.config(state="normal", text="🛑 Stop")
    threading.Thread(target=record_stream, daemon=True).start()

def record_stream():
    global frames
    with sd.InputStream(samplerate=fs, channels=1, dtype='float32', callback=callback):
        while recording:
            sd.sleep(100)

def callback(indata, frames_count, time, status):
    frames.append(indata.copy())

def stop_recording():
    global recording
    if not recording:
        return
    recording = False
    start_button.config(state="normal")
    stop_button.config(state="disabled", text="🛑 Stop")
    
    # Combine frames and save with unique filename
    audio = np.concatenate(frames, axis=0)
    filename = f"temp_{int(time.time())}.wav"
    wavio.write(filename, audio, fs, sampwidth=2)
    
    # Transcribe with Whisper
    result = model.transcribe(filename)
    print("Transcription:", result["text"])
    text = result["text"].lower()
    text = words_to_math(text)
    entry.delete(0, tk.END)
    entry.insert(0, text)
    text = text.replace(',', '')
    text = re.sub(r'\.(?=\s|$)', '', text)
    text = re.sub(r",(?=[^()]*\))", "", text) 
    text = re.sub(r",", " ", text)
    calculate()

#--- Button callback functions ---
def button_cal():
    text = entry.get()
    print("Original input:", text)


    calculate()

def toggle_radians():
    global use_radians
    use_radians = not use_radians
    print("Use radians:", use_radians)

#--- Text conversion functions ---
def clean_for_sympy(expr):
    expr = expr.replace("math.", "")
    expr = re.sub(r'pow\(([^,]+),([^)]+)\)', r'(\1)**(\2)', expr)
    return expr

def parentheses_balance(text):
    """Label spoken 'open'/'close' tokens with unique pair ids.

    This function replaces occurrences of the words `open` and `close` with
    indexed tokens like `(_1` and `)_1` so downstream handlers can reliably
    locate matching pairs even in complex or nested inputs.

    Behavior notes:
    - Uses a stack to assign unique ids to matching open/close pairs.
    - Unmatched closes are labeled `)_0` to make them visible for later cleanup.

    Example:
      'open open x close close' -> '(_1 (_2 x )_2 )_1'
    """
    # Use a stack with unique sequential ids for each open/close pair so that
    # pairs don't get reused when parentheses appear at the same nesting depth
    pattern = re.compile(r'\b(open|close|closed)\b', flags=re.IGNORECASE)
    pos = 0
    stack = []  # holds ids of currently open parentheses
    next_id = 1
    while True:
        m = pattern.search(text, pos)
        if not m:
            break
        token = m.group(1)
        if token == "open":
            # Assign a new id to this open and push it
            label = f"(_{next_id}"
            stack.append(next_id)
            next_id += 1
            text = text[:m.start()] + label + text[m.end():]
            pos = m.start() + len(label)
            print(f"Found 'open': assigned id {stack[-1]}, stack now: {stack}")
        else:  # close
            # Pop the matching open id if present, otherwise label as unmatched
            if stack:
                rid = stack.pop()
                label = f")_{rid}"
                print(f"Found 'close': matched id {rid}, stack now: {stack}")
            else:
                # unmatched close -> label with 0 so it remains visible
                label = ")_0"
                print("Found unmatched 'close': labeled as )_0")
            text = text[:m.start()] + label + text[m.end():]
            pos = m.start() + len(label)
    return text

# --- Parentheses helpers ---
def get_paren_pairs(text):
    print("Getting paren pairs for text:", text)
    """Return a dict mapping index -> (open_start, open_end, close_start, close_end).
    Assumes text contains tokens like '(_N' and ')_N'. Positions are string indices (start/end).
    """
    opens = {int(m.group(1)): (m.start(), m.end()) for m in re.finditer(r'\(_(\d+)', text)}
    print("Opens found:", opens)
    closes = {int(m.group(1)): (m.start(), m.end()) for m in re.finditer(r'\)_(\d+)', text)}
    print("Closes found:", closes)

    pairs = {}
    for n in set(list(opens.keys()) + list(closes.keys())):
        o = opens.get(n, (None, None))
        c = closes.get(n, (None, None))
        pairs[n] = (o[0], o[1], c[0], c[1])
    return pairs

def get_paren_content(text, n):
    print("Getting paren content for n =", n, "in text:", text)
    """Return the substring contained inside the indexed parentheses `n`.

    Returns None if the pair is missing or unbalanced. This is a safe helper
    used by handlers that need to extract the content of a parenthesised phrase.
    """
    pairs = get_paren_pairs(text)
    if n not in pairs:
        return None
    open_start, open_end, close_start, close_end = pairs[n]
    if open_end is None or close_start is None:
        return None
    return text[open_end:close_start]

def fix_unmatched_parens(text):
    print("Fixing unmatched parens in text:", text)
    """Remove stray closing parens and append missing closing parens at end.

    This mirrors the balancing used in `words_to_math` but is available
    as a reusable helper for other routines that produce intermediate
    expressions (e.g., derivative/integral handling) before final cleanup.
    """
    out = []
    bal = 0
    for ch in text:
        if ch == '(':
            bal += 1
            out.append(ch)
            print("Found '(': bal =", bal)
        elif ch == ')':
            if bal == 0:
                # skip unmatched closing paren
                continue
            bal -= 1
            out.append(ch)
            print("Found ')': bal =", bal)
        else:
            out.append(ch)
            print("Found other char:", ch)
    if bal > 0:
        out.append(')' * bal)
        print("Appending", bal, "closing parens at end")
    return ''.join(out)

def parse_substitution(raw_text):
    """Parse simple substitution directives from spoken/raw text.

    Returns a tuple (u_var, u_expr_str) if a substitution like
    'let u = x squared' or 'substitute u = x^2' is found, otherwise None.
    """
    text = raw_text.lower()
    # Common patterns: 'let u = ...', 'substitute u = ...', 'u substitution u = ...'
    # Try to find explicit 'let'/'substitute' patterns capturing the rest of the line as RHS
    m = re.search(r"\b(?:let|substitute|use substitution|using substitution|u-substitution)\s+([a-zA-Z])\s*(?:=|equals?|equal)\s*(.+)$", text)
    if not m:
        # fallback: standalone 'u = <expr>' but prefer common substitution letters
        m2 = re.search(r"\b([a-zA-Z])\s*(?:=|equals?|equal)\s*(.+)$", text)
        if m2 and m2.group(1).lower() in ('u', 't', 'v'):
            m = m2

    if not m:
        return None

    u = m.group(1).strip()
    rhs_start = m.start(2)
    # skip leading whitespace
    while rhs_start < len(text) and text[rhs_start].isspace():
        rhs_start += 1

    # Case: labeled parentheses '(_N' — use get_paren_pairs
    if text[rhs_start:rhs_start+2] == '(_':
        m_label = re.match(r"\(_(\d+)", text[rhs_start:])
        if m_label:
            n = int(m_label.group(1))
            pairs = get_paren_pairs(text)
            if n in pairs and pairs[n][2] is not None:
                o_s, o_e, c_s, c_e = pairs[n]
                content = text[o_e:c_s].strip()
                print(f"Parsed substitution (labeled paren): {u} = {content}")
                return (u, content)

    # Case: normal parentheses — reuse paren-labeling to find match robustly
    if rhs_start < len(text) and text[rhs_start] == '(':
        rhs_sub = text[rhs_start:]
        labeled = parentheses_balance(rhs_sub)
        pairs_local = get_paren_pairs(labeled)
        candidates = sorted([(n, data) for n, data in pairs_local.items() if data[0] is not None and data[2] is not None], key=lambda x: x[1][0])
        if candidates:
            n1 = candidates[0][0]
            o_s, o_e, c_s, c_e = pairs_local[n1]
            content = labeled[o_e:c_s].strip()
            print(f"Parsed substitution (paren via labeling): {u} = {content}")
            return (u, content)

    # Fallback: take until delimiter or end
    rhs = text[rhs_start:].strip()
    rhs = re.split(r"\b(?:du|dx|dt|,|;|and|from)\b", rhs)[0].strip()
    rhs = rhs.rstrip(') ').strip()

    print(f"Parsed substitution (fallback): {u} = {rhs}")
    return (u, rhs)

def find_matching_close_for_open(text, open_pos):
    """Given an index pointing to an '(_N' token, return the start index of ')_N'.

    Useful when code has the position of an open token and wants to quickly
    jump to its matching close. Returns None if the token at `open_pos` is not an
    '(_N' or if the matching close is missing.
    """
    m = re.match(r'\(_(\d+)', text[open_pos:open_pos+10])
    if not m:
        return None
    n = int(m.group(1))
    pairs = get_paren_pairs(text)
    if n in pairs and pairs[n][2] is not None:
        return pairs[n][2]
    return None

def Absolute_value(text):
    # Handle labeled parentheses: 'absolute value of (_N ... )_N' -> 'abs(content)'
    pairs = get_paren_pairs(text)
    for m in re.finditer(r"\b(absolute value of|the absolute value of|the magnitude of|absolute value|the magnitude)\b", text):
        start = m.end()
        candidates = sorted([(n, data) for n, data in pairs.items() if data[0] is not None and data[0] > start], key=lambda x: x[1][0])
        if not candidates:
            continue
        n1 = candidates[0][0]
        o_s, o_e, c_s, c_e = pairs[n1]
        if o_e is None or c_s is None:
            continue
        content = text[o_e:c_s]
        replace_start = m.start()
        replace_end = c_e
        text = text[:replace_start] + f"abs({content})" + text[replace_end:]
        pairs = get_paren_pairs(text)

    # Convert phrases like 'absolute value of X' to 'abs(X)'
    text = re.sub(r"absolute value of (-?[A-Za-z0-9]+)", r"abs(\1", text)
    text = re.sub(r"absolute value(-?[A-Za-z0-9]+)", r"abs(\1", text)
    text = re.sub(r"the absolute value of (-?[A-Za-z0-9]+)", r"abs(\1", text)
    text = re.sub(r"the absolute value(-?[A-Za-z0-9]+)", r"abs(\1", text)
    text = re.sub(r"the magnitude of (-?[A-Za-z0-9]+)", r"abs(\1", text)
    text = re.sub(r"the magnitude(-?[A-Za-z0-9]+)", r"abs(\1", text)

    return text

def handle_trig_degrees(text):

    global use_radians

    # First, handle indexed parentheses like 'sin of (_N ... )_N'
    pairs = get_paren_pairs(text)
    for m in re.finditer(r"\b(cos(?:ine)?|sin(?:e)?|tan(?:gent)?)\s+of\b", text):
        func = m.group(1)
        start = m.end()
        candidates = sorted([(n, data) for n, data in pairs.items() if data[0] is not None and data[0] > start], key=lambda x: x[1][0])
        if not candidates:
            continue
        n1 = candidates[0][0]
        o_s, o_e, c_s, c_e = pairs[n1]
        if o_e is None or c_s is None:
            continue
        content = text[o_e:c_s].strip()
        if use_radians:
            replacement = f"math.{func}({content})"
        else:
            replacement = f"math.{func}(math.radians({content})"
        text = text[:m.start()] + replacement + text[c_e:]
        pairs = get_paren_pairs(text)

    # Fallback: original worded 'open ... close' patterns
    if use_radians:
        text = re.sub(r"cos(?:ine)? of \s+open\s+(.*?)\s+close", r"math.cos(\1)", text)
        text = re.sub(r"sin(?:e)? of \s+open\s+(.*?)\s+close", r"math.sin(\1)", text)
        text = re.sub(r"tan(?:gent)? of \s+open\s+(.*?)\s+close", r"math.tan(\1)", text)
    else:
        text = re.sub(r"cos(?:ine)? of \s+open\s+(.*?)\s+close", r"math.cos(math.radians(\1)", text)
        text = re.sub(r"sin(?:e)? of \s+open\s+(.*?)\s+close", r"math.sin(math.radians(\1)", text)
        text = re.sub(r"tan(?:gent)? of \s+open\s+(.*?)\s+close", r"math.tan(math.radians(\1)", text)
    return text

def handle_inverse_trig(text):
    """Handle inverse trig phrases like 'arcsin of (_N ... )_N' and 'inverse sine of open ... close'.

    If `use_radians` is False then converts the result to degrees by multiplying by 180/math.pi.
    """
    global use_radians

    pairs = get_paren_pairs(text)

    # arcsin / inverse sine
    for m in re.finditer(r"\b(arcsin|arc sine|asin|inverse sine|inverse of sine)\b", text):
        start = m.end()
        candidates = sorted([(n, data) for n, data in pairs.items() if data[0] is not None and data[0] > start], key=lambda x: x[1][0])
        if not candidates:
            continue
        n1 = candidates[0][0]
        o_s, o_e, c_s, c_e = pairs[n1]
        if o_e is None or c_s is None:
            continue
        content = text[o_e:c_s].strip()
        if use_radians:
            replacement = f"math.asin({content})"
        else:
            replacement = f"(math.asin({content})*180/math.pi)"
        text = text[:m.start()] + replacement + text[c_e:]
        pairs = get_paren_pairs(text)

    # arccos / inverse cosine
    for m in re.finditer(r"\b(arccos|arc cosine|acos|inverse cosine|inverse of cosine)\b", text):
        start = m.end()
        candidates = sorted([(n, data) for n, data in pairs.items() if data[0] is not None and data[0] > start], key=lambda x: x[1][0])
        if not candidates:
            continue
        n1 = candidates[0][0]
        o_s, o_e, c_s, c_e = pairs[n1]
        if o_e is None or c_s is None:
            continue
        content = text[o_e:c_s].strip()
        if use_radians:
            replacement = f"math.acos({content})"
        else:
            replacement = f"(math.acos({content})*180/math.pi)"
        text = text[:m.start()] + replacement + text[c_e:]
        pairs = get_paren_pairs(text)

    # arctan / inverse tangent
    for m in re.finditer(r"\b(arctan|arc tangent|atan|inverse tangent|inverse of tangent)\b", text):
        start = m.end()
        candidates = sorted([(n, data) for n, data in pairs.items() if data[0] is not None and data[0] > start], key=lambda x: x[1][0])
        if not candidates:
            continue
        n1 = candidates[0][0]
        o_s, o_e, c_s, c_e = pairs[n1]
        if o_e is None or c_s is None:
            continue
        content = text[o_e:c_s].strip()
        if use_radians:
            replacement = f"math.atan({content})"
        else:
            replacement = f"(math.atan({content})*180/math.pi)"
        text = text[:m.start()] + replacement + text[c_e:]
        pairs = get_paren_pairs(text)

    # Fallback worded forms
    if use_radians:
        text = re.sub(r"(arcsin|arc ?sine|asin|inverse sine) of open\s+(.*?)\s+close", r"math.asin(\2)", text)
        text = re.sub(r"(arccos|arc ?cosine|acos|inverse cosine) of open\s+(.*?)\s+close", r"math.acos(\2)", text)
        text = re.sub(r"(arctan|arc ?tangent|atan|inverse tangent) of open\s+(.*?)\s+close", r"math.atan(\2)", text)
    else:
        text = re.sub(r"(arcsin|arc ?sine|asin|inverse sine) of open\s+(.*?)\s+close", r"(math.asin(\2)*180/math.pi)", text)
        text = re.sub(r"(arccos|arc ?cosine|acos|inverse cosine) of open\s+(.*?)\s+close", r"(math.acos(\2)*180/math.pi)", text)
        text = re.sub(r"(arctan|arc ?tangent|atan|inverse tangent) of open\s+(.*?)\s+close", r"(math.atan(\2)*180/math.pi)", text)

    # Additional simple fallbacks: 'arcsin x', 'arccos x', 'arctan x', 'asin x', etc.
    if use_radians:
        text = re.sub(r"\b(arcsin|asin|arc ?sine|inverse sine)\s+(.*?)\s+close", r"math.asin(\2)", text)
        text = re.sub(r"\b(arccos|acos|arc ?cosine|inverse cosine)\s+(.*?)\s+close", r"math.acos(\2)", text)
        text = re.sub(r"\b(arctan|atan|arc ?tangent|inverse tangent)\s+(.*?)\s+close", r"math.atan(\2)", text)
    else:
        text = re.sub(r"\b(arcsin|asin|arc ?sine|inverse sine)\s+(.*?)\s+close", r"(math.asin(\2)*180/math.pi)", text)
        text = re.sub(r"\b(arccos|acos|arc ?cosine|inverse cosine)\s+(.*?)\s+close", r"(math.acos(\2)*180/math.pi)", text)
        text = re.sub(r"\b(arctan|atan|arc ?tangent|inverse tangent)\s+(.*?)\s+close", r"(math.atan(\2)*180/math.pi)", text)

    return text

def handle_power_phrases(text):
    """Convert spoken power phrases into math.* forms.

    Strategy:
    - First operate on indexed parentheses produced by `parentheses_balance` so
      we can robustly find matching opens and closes without ambiguity.
    - For phrases like 'to the power of' and 'raised to' we process matches
      right-to-left (innermost first) to avoid interfering with nested exponents.
    - Fallbacks handle older 'open ... close' worded forms.
    """
    # Handle labeled parentheses first (e.g., '(_N ... )_N squared')
    pairs = get_paren_pairs(text)

    # squared / cube after a pair
    for n, (o_s, o_e, c_s, c_e) in sorted(pairs.items(), key=lambda x: x[1][0] if x[1][0] is not None else 0):
        if o_e is None or c_s is None:
            continue
        after = text[c_e:c_e+20]
        if re.match(r"\s*squared\b", after):
            content = text[o_e:c_s]
            replace_start = o_s
            replace_end = c_e + re.search(r"\s*squared\b", after).end()
            text = text[:replace_start] + f"math.pow({content}, 2)" + text[replace_end:]
            pairs = get_paren_pairs(text)
        elif re.match(r"\s*cube\b", after):
            content = text[o_e:c_s]
            replace_start = o_s
            replace_end = c_e + re.search(r"\s*cube\b", after).end()
            text = text[:replace_start] + f"math.pow({content}, 3)" + text[replace_end:]
            pairs = get_paren_pairs(text)

    # Handle 'A to the power of B' using pairs around 'to the power of'
    matches = list(re.finditer(r"\bto the power of\b", text))
    # Process from right-to-left so inner exponents are handled before outer ones
    for m in sorted(matches, key=lambda mm: mm.start(), reverse=True):
        start = m.start()
        # find previous pair (base)
        pairs = get_paren_pairs(text)
        prev = sorted([(n, data) for n, data in pairs.items() if data[2] is not None and data[2] < start], key=lambda x: x[1][2])
        if not prev:
            continue
        n_base = prev[-1][0]
        base = pairs[n_base]
        # find next pair (exponent)
        nexts = sorted([(n, data) for n, data in pairs.items() if data[0] is not None and data[0] > m.end()], key=lambda x: x[1][0])
        if not nexts:
            continue
        n_arg = nexts[0][0]
        arg = pairs[n_arg]
        base_content = text[base[1]:base[2]]
        arg_content = text[arg[1]:arg[2]]
        replace_start = base[0]
        replace_end = arg[3]
        text = text[:replace_start] + f"math.pow({base_content}, {arg_content})" + text[replace_end:]
        # Recompute pairs after mutation
        pairs = get_paren_pairs(text)

    # Handle 'raised to' / 'raised to the power of' similarly
    matches = list(re.finditer(r"\braised(?: to the power of| to)\b", text))
    for m in sorted(matches, key=lambda mm: mm.start(), reverse=True):
        start = m.start()
        pairs = get_paren_pairs(text)
        prev = sorted([(n, data) for n, data in pairs.items() if data[2] is not None and data[2] < start], key=lambda x: x[1][2])
        if not prev:
            continue
        n_base = prev[-1][0]
        base = pairs[n_base]
        nexts = sorted([(n, data) for n, data in pairs.items() if data[0] is not None and data[0] > m.end()], key=lambda x: x[1][0])
        if not nexts:
            continue
        n_arg = nexts[0][0]
        arg = pairs[n_arg]
        base_content = text[base[1]:base[2]]
        arg_content = text[arg[1]:arg[2]]
        replace_start = base[0]
        replace_end = arg[3]
        text = text[:replace_start] + f"math.pow({base_content}, {arg_content})" + text[replace_end:]
        pairs = get_paren_pairs(text)
    # square root patterns using pairs
    for m in re.finditer(r"\bsquare root(?: of)?\b", text):
        start = m.end()
        pairs = get_paren_pairs(text)
        candidates = sorted([(n, data) for n, data in pairs.items() if data[0] is not None and data[0] >= start], key=lambda x: x[1][0])
        if not candidates:
            continue
        n1 = candidates[0][0]
        o_s, o_e, c_s, c_e = pairs[n1]
        content = text[o_e:c_s]
        replace_start = m.start()
        replace_end = c_e
        text = text[:replace_start] + f"math.sqrt({content})" + text[replace_end:]
        pairs = get_paren_pairs(text)

    # Fallback: original open...close regexes
    text = re.sub(r"open\s+(.*?)\s+close\s+squared\b",r"math.pow(\1, 2)",text)
    text = re.sub(r"open\s+(.*?)\s+close\s+cube\b",r"math.pow(\1, 3)",text)

    # --- A to the power of B ---
    text = re.sub(r"open\s+(.*?)\s+close\s+to the power of\s+open\s+(.*?)\s+close",r"(\1)**(\2)",text)
    text = re.sub(r"open\s+(.*?)\s+close\s+tothepowerof\s+open\s+(.*?)\s+close", r"math.pow(\1, \2)", text)

    # --- A to the power B ---
    text = re.sub(r"open\s+(.*?)\s+close\s+to the power\s+open\s+(.*?)\s+close", r"math.pow(\1, \2)", text)
    text = re.sub(r"open\s+(.*?)\s+close\s+tothepower\s+open\s+(.*?)\s+close", r"math.pow(\1, \2)", text)
    
    # --- A raised to the power of B ---
    text = re.sub(r"open\s+(.*?)\s+close\s+raised to the power of\s+open\s+(.*?)\s+close", r"math.pow(\1, \2)", text)
    text = re.sub(r"open\s+(.*?)\s+close\s+raisedtothepowerof\s+open\s+(.*?)\s+close", r"math.pow(\1, \2)", text)
    # --- A raised to B ---
    text = re.sub(r"open\s+(.*?)\s+close\s+raised to\s+\s+open\s+(.*?)\s+close", r"math.pow(\1, \2)", text)
    text = re.sub(r"open\s+(.*?)\s+close\s+raisedto\s+\s+open\s+(.*?)\s+close", r"math.pow(\1, \2)", text)

    # --- A power B ---
    text = re.sub(r"open\s+(.*?)\s+close\s+power\s+\s+open\s+(.*?)\s+close", r"math.pow(\1, \2", text)

    
    # --- A square root ---
    text = re.sub(r"square root of\s+open\s+(.*?)\s+close", r"math.sqrt(\1)", text)
    text = re.sub(r"squarerootof\s+open\s+(.*?)\s+close", r"math.sqrt(\1)", text)
    text = re.sub(r"square root\s+open\s+(.*?)\s+close", r"math.sqrt(\1)", text)
    text = re.sub(r"squareroot\s+open\s+(.*?)\s+close", r"math.sqrt(\1)", text)
    text = re.sub(r"square root of negative (\d+)", r"math.sqrt(-\1)", text)

    # --- nth root of A (parenthesized and numeric) ---
    text = re.sub(r"(\d+)(?:st|nd|rd|th)? root of\s+open\s+(.*?)\s+close", r"math.pow(\2, 1/\1)", text)
    # numeric '3rd root of 8' -> math.pow(8, 1/3)
    text = re.sub(r"(\d+)(?:st|nd|rd|th)? root of\s+(\d+(?:\.\d+)?)\b", r"math.pow(\2, 1/\1)", text)
    # 'root of 9' -> sqrt(9)
    text = re.sub(r"\broot of\s+(\d+(?:\.\d+)?)\b", r"math.sqrt(\1)", text)

    text = re.sub(r"root\s+open\s+(.*?)\s+close", r"math.sqrt(\1)", text)


    return text

def handle_fractions(text):
    # number words → digits
    words = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9"
    }

    # one half, three fourths, etc.
    for wnum, d in words.items():
        for wden, d2 in words.items():
            text = re.sub(rf"{wnum} (?:over|out of) {wden}", f"{d}/{d2}", text)
            text = re.sub(rf"{wnum} (?:over|outof) {wden}", f"{d}/{d2}", text)
            text = re.sub(rf"{wnum} {wden}(?:s|th|ths)?", f"{d}/{d2}", text)

    # Handle labeled parentheses: '(_N a )_N over (_M b )_M' -> 'a/b'
    pairs = get_paren_pairs(text)
    for m1, (o1_s, o1_e, c1_s, c1_e) in sorted(pairs.items(), key=lambda x: x[1][0] if x[1][0] is not None else 0):
        if c1_e is None:
            continue
        # look for 'over' after c1_e
        over_match = re.search(r"\bover\b", text[c1_e:])
        if not over_match:
            continue
        pos_over = c1_e + over_match.start()
        # find next pair after 'over'
        candidates = sorted([(n, data) for n, data in pairs.items() if data[0] is not None and data[0] > pos_over], key=lambda x: x[1][0])
        if not candidates:
            continue
        m2 = candidates[0][0]
        o2_s, o2_e, c2_s, c2_e = pairs[m2]
        if o2_e is None or c2_s is None:
            continue
        a = text[o1_e:c1_s]
        b = text[o2_e:c2_s]
        replace_start = o1_s
        replace_end = c2_e
        text = text[:replace_start] + f"{a}/{b}" + text[replace_end:]
        pairs = get_paren_pairs(text)

    # numeric fractions: 3 over 8 (fallback)
    text = re.sub(r"open\s+(.*?)\s+close\s+over\s+\s+open\s+(.*?)\s+close", r"\1/\2", text)

    return text

def handle_percentages(text):
    # Handle labeled parentheses: '(_N x )_N percent' -> '(x/100)'
    pairs = get_paren_pairs(text)
    for n, (o_s, o_e, c_s, c_e) in sorted(pairs.items(), key=lambda x: x[1][0] if x[1][0] is not None else 0):
        if o_e is None or c_s is None:
            continue
        after = text[c_e:c_e+20]
        if re.match(r"\s*percent\b", after) or re.match(r"\s*per\s*cent\b", after):
            content = text[o_e:c_s]
            replace_start = o_s
            # include the percent words
            perc_m = re.search(r"\s*percent\b|\s*per\s*cent\b", after)
            replace_end = c_e + (perc_m.end() if perc_m else 0)
            text = text[:replace_start] + f"({content}/100)" + text[replace_end:]
            pairs = get_paren_pairs(text)

    # 20 percent → (20/100)
    text = re.sub(r"open\s+(.*?)\s+close\s+percent", r"(\1/100)", text)

    return text

def handle_logs(text):
    # --- Simple numeric forms: "log base 2 of 8" ---
    text = re.sub(r"log base\s+(\d+)\s+of\s+(\d+)", r"math.log(\2, \1)", text)

    # --- Handle indexed parentheses produced by parentheses_balance: '(_N ... )_N' ---
    # Use get_paren_pairs/get_paren_content to find the base and argument pairs
    pairs = get_paren_pairs(text)

    # Replace occurrences of 'log base <paren> of <paren>'
    offset = 0
    for m in re.finditer(r"\blog base\b", text):
        start = m.start()
        # Find the first pair after 'log base'
        candidates = sorted([(n, data) for n, data in pairs.items() if data[0] is not None and data[0] > start], key=lambda x: x[1][0])
        if not candidates:
            continue
        n_base = candidates[0][0]
        base_open_s, base_open_e, base_close_s, base_close_e = pairs[n_base]

        # Find 'of' after the base close
        of_match = re.search(r"\bof\b", text[base_close_e:])
        if not of_match:
            continue
        of_pos = base_close_e + of_match.start()

        # Find the next pair after 'of'
        candidates2 = sorted([(n, data) for n, data in pairs.items() if data[0] is not None and data[0] > of_pos], key=lambda x: x[1][0])
        if not candidates2:
            continue
        n_arg = candidates2[0][0]
        arg_open_s, arg_open_e, arg_close_s, arg_close_e = pairs[n_arg]

        base_content = text[base_open_e:base_close_s]
        arg_content = text[arg_open_e:arg_close_s]

        # Replace the whole span 'log base <base_pair> of <arg_pair>' with math.log(arg, base)
        replace_start = start
        replace_end = arg_close_e
        replacement = f"math.log({arg_content}, {base_content})"
        text = text[:replace_start] + replacement + text[replace_end:]

        # Recompute pairs because indices changed
        pairs = get_paren_pairs(text)

    # --- Natural log / ln ---
    # numeric case: 'ln of 8' or 'natural log of 8'
    text = re.sub(r"\b(natural log of|ln of|ln)\s+(\d+)\b", r"log(\2, e)", text)

    # labeled paren case: 'ln of (_N ... )_N'
    pairs = get_paren_pairs(text)
    for m in re.finditer(r"\b(natural log of|ln of|ln)\b", text):
        start = m.end()
        # find first pair after this position
        candidates = sorted([(n, data) for n, data in pairs.items() if data[0] is not None and data[0] > start], key=lambda x: x[1][0])
        if not candidates:
            continue
        n1 = candidates[0][0]
        o_s, o_e, c_s, c_e = pairs[n1]
        content = text[o_e:c_s]
        replace_start = m.start()
        replace_end = c_e
        replacement = f"log({content}, e)"
        text = text[:replace_start] + replacement + text[replace_end:]
        pairs = get_paren_pairs(text)

    # --- Fall-back patterns for worded 'open ... close' (pre-label format) ---
    text = re.sub(r"log base open\s+(.*?)\s+close of\s+open\s+(.*?)\s+close", r"math.log(\2, \1)", text)
    text = re.sub(r"logbaseopen\s+(.*?)\s+closeof\s+open\s+(.*?)\s+close", r"math.log(\2, \1", text)
    text = re.sub(r"log base open\s+(.*?)\s+close*of \s+open\s+(.*?)\s+close", r"math.log(\2, \1)", text)

    text = re.sub(r"(natural log of|ln of|ln)\s+open\s+(.*?)\s+close", r"log(\2, e)", text)
    text = re.sub(r"(naturallogof|lnof)\s+open\s+(.*?)\s+close", r"log(\2, e)", text)

    return text

def convert_to_sympy(expr_str):
    global use_radians

    # --- Remove leading/trailing * and spaces ---
    expr_str = re.sub(r'^\*+', '', expr_str)   # leading *
    expr_str = re.sub(r'\*+$', '', expr_str)   # trailing *

    # Remove duplicate 'math.' if present
    #expr_str = re.sub(r'(math\.)(math\.)+', r'\1', expr_str)
    #expr_str = re.sub(r'(math\.)(math\.)(math\.)+', r'\1', expr_str)

    # --- Replace math functions with sympy ---
    expr_str = expr_str.replace("math.sin", "sin").replace("math.cos", "cos").replace("math.tan", "tan").replace("math.asin", "asin").replace("math.acos", "acos").replace("math.atan", "atan")
    expr_str = expr_str.replace("mathsqrt", "sqrt").replace("math.sqrt", "sqrt")
    expr_str = expr_str.replace("math.log", "log").replace("mathlog", "log")
    expr_str = expr_str.replace("math.pi", "pi").replace("math.e", "E").replace("mathpi", "pi").replace("mathe", "E")
    expr_str = expr_str.replace("mathpow", "pow").replace("math.pow", "pow")
    expr_str = re.sub(r'\b(sin|cos|tan)\s+([a-zA-Z0-9_]+)\b', r'\1(\2)', expr_str)


    # --- Convert trig from degrees if needed ---
    if not use_radians:
        expr_str = re.sub(r"sin\(([^)]+)\)", r"sin(\1*pi/180)", expr_str)
        expr_str = re.sub(r"cos\(([^)]+)\)", r"cos(\1*pi/180)", expr_str)
        expr_str = re.sub(r"tan\(([^)]+)\)", r"tan(\1*pi/180)", expr_str)

    # --- Powers: convert pow(a,b) → (a**b) ---
    expr_str = re.sub(r"pow\(([^,]+),([^)]+)\)", r"(\1**\2)", expr_str)

    # ---- Absolute value ---
    expr_str = re.sub(r"abs\(([^)]+)\)", r"Abs(\1)", expr_str)

    # --- Imaginary unit ---
    expr_str = re.sub(r"(\d+)\s*i", r"\1*I", expr_str)  # 3i → 3*I
    expr_str = re.sub(r"\bi\b", "I", expr_str)          # standalone i → I
    expr_str = re.sub(r"sqrt\(-1\)", "I", expr_str)
    expr_str = re.sub(r"i squared", "I**2", expr_str)

    # --- Implicit multiplication fixes ---
    expr_str = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", expr_str)     # 3x → 3*x
    expr_str = re.sub(r"([a-zA-Z])(\d)", r"\1*\2", expr_str)     # x3 → x*3
    expr_str = re.sub(r"(\d)\s*\(", r"\1*(", expr_str)           # 3(x+1) → 3*(x+1)
    expr_str = re.sub(r"\)\s*([a-zA-Z0-9])", r")*\1", expr_str)  # (x)y → (x)*y
    expr_str = re.sub(r"\)\s*\(", r")*(", expr_str)              # (x)(y) → (x)*(y)
    
    # Only insert * between a number/variable and a known constant or single-letter variable
    expr_str = re.sub(r'(\d+|\b[a-zA-Z]\b)\s*(pi|E|I)\b', r'\1*\2', expr_str)

    # --- Clean up spaces inside parentheses ---
    expr_str = re.sub(r'\(\s+', '(', expr_str)
    expr_str = re.sub(r'\s+\)', ')', expr_str)

    # --- Fractions like 1/2 y → 1/2*y ---
    expr_str = re.sub(r'(\d+/\d+)\s*([a-zA-Z])', r'\1*\2', expr_str)

    #-----E------
    expr_str = re.sub(r'\b-math\.e\b', '-E', expr_str)
    expr_str = re.sub(r'math\.e\b', 'E', expr_str)
    
    #----- Infinity -----
    expr_str = re.sub(r'\b-math.inf\b', '-oo', expr_str)
    expr_str = re.sub(r'\bmath.inf\b', 'oo', expr_str)
    return expr_str.strip()

def words_to_math(text):
    #Top-level pipeline to convert spoken-word math into a Python/math string.

    #Pipeline stages (in order):
    #1) normalize text and label parentheses with `parentheses_balance`
    #2) trig handling (degrees/radians)
    #3) power phrases (squared, nth root, 'to the power of')
    #4) fractions, percentages
    #5) logs and natural logs
    #6) absolute value
    #7) convert labeled parentheses back to normal '(' and ')' and replace
       #common operator words with symbols
    #8) final cleanup: normalize 'square root of' and ensure balanced parens

    #Set DEBUG=True at the top of the file to see step-by-step prints.
    

    text = text.lower()
    text = text.rstrip(" .,!?:;")
    text = re.sub(r'\.(?=\s|$)', '', text)
    if DEBUG:
        print("Striped text: ", text)

    # Convert 'negative X' → '-X'
    text = re.sub(r"negative (\d+)", r"-\1", text)
    text = re.sub(r"negative (\w+)", r"-\1", text)  # also works with 'negative one'

    # Check for balanced parentheses
    text = parentheses_balance(text)
    print("After parentheses balance:", text)
    # 1. Trig first
    text = handle_trig_degrees(text)
    print("After handle_trig_degrees:", text)
    # 1b. Inverse trig
    text = handle_inverse_trig(text)
    print("After handle_inverse_trig:", text)
    # 2. Powers
    text = handle_power_phrases(text)
    print("After handle_power_phrases:", text)
    # 3. Fractions
    text = handle_fractions(text)
    print("After handle_fractions:", text)
    # 4. Percentages
    text = handle_percentages(text)
    print("After handle_percentages:", text)
    # 5. Logs
    text = handle_logs(text)
    print("After handle_logs:", text)
    # 6. Absolute value
    text = Absolute_value(text)
    print("After Absolute_value:", text)

    # Convert labeled parentheses like '(_N' and ')_N' back to normal '(' and ')'
    text = re.sub(r"\(_\d+", "(", text)
    text = re.sub(r"\)_\d+", ")", text)
    print("After removing labeled parentheses:", text)

    # 7. Replace word operators
    replacements = {
        "plus": "+",
        "minus": "-",
        "times": "*",
        "multiplied by": "*",
        "divided by": "/",
        "over": "/",
    }
    for word, symbol in replacements.items():
        text = text.replace(word, symbol)

    # 8. Math constants
    text = re.sub(r"\bpi\b", "math.pi", text)
    text = re.sub(r"\be\b", "math.e", text)
    text = re.sub(r"\bequals\b", "=", text)
    text = re.sub(r"\bequal\b", "=", text)


    # 9. Imaginary unit
    text = re.sub(r"(\d+)\s*i", r"\1*I", text)  # 4i → 4*I
    text = re.sub(r"\bi\b", "I", text)          # standalone i → I
    text = re.sub(r"sqrt\(-1\)", "I", text)     # sqrt(-1) → I
    text = re.sub(r"i squared", "I**2", text) 



    # 10. Implicit multiplication
    # Between numbers and variables
    text = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", text)  # 3x → 3*x
    text = re.sub(r"([a-zA-Z])(\d)", r"\1*\2", text)  # x3 → x*3
    # Between number and parentheses
    text = re.sub(r"(\d)\s*\(", r"\1*(", text)        # 3(x+1) → 3*(x+1)
    # Between parentheses and number or variable
    text = re.sub(r"\)\s*([a-zA-Z0-9])", r")*\1", text) # (3)x → (3)*x, (2+1)5 → (2+1)*5
    # Between a closing parenthesis and a math.* function (e.g., ') math.sqrt' -> ')*math.sqrt')
    text = re.sub(r"\)\s*(math\.[a-zA-Z_\.]+)", r")*\1", text)
    # Between a closing parenthesis and a bare function like 'sqrt' (') sqrt' -> ')*sqrt')
    text = re.sub(r"\)\s*(sqrt\b)", r")*\1", text)
    # Between two sets of parentheses
    text = re.sub(r"\)\s*\(", r")*(", text)           # (x+1)(y+2) → (x+1)*(y+2)
    # Between variable/number and math constant
    text = re.sub(r"([0-9a-zA-Z])\s*(math\.[a-zA-Z_]+)", r"\1*\2", text)
    # Remove spaces right after '(' and before ')'
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    # Make sure any fraction followed by variable has explicit multiplication
    text = re.sub(r'(\d+/\d+)\s*([a-zA-Z])', r'\1*\2', text)
    # fix accidental leading *
    text = re.sub(r'^\*+', '', text)
    # infinity
    text = re.sub(r'\b(infinity|inf)\b', 'math.inf', text)
    text = re.sub(r'\b-negative infinity\b', '-math.inf', text)

    text = re.sub(r'(math\.)(math\.)+', r'\1', text)



    return text.strip()

def to_python_number(val):
    # Convert sympy Zero, Integer, Rational, Float, etc.
    try:
        return float(val)
    except Exception:
        return complex(val)

#------Simplifaction-------
def Simplify_expression(expression):
    try:
        print("Symbolic simplify requested")

        expr_str = expression.lower()
        expr_str = re.sub(r"(simplify|simplified|simplification)", "", expr_str)
        expr_str = expr_str.strip()

        expr_sym = sympify(convert_to_sympy(expr_str))
        print("Right before simplify", expr_sym)

        result = simplify(expr_sym)
        print("Simplified result:", result)
        entry.settext(str(result))

        output_label.config(text=f"Simplified:\n{result}")
        return

    except Exception as e:
        output_label.config(text=f"Simplify error: {e}")
        return  

#-----rounding functions -----
def round_complex_number(c, decimals=5):
    """Round a complex number (SymPy) to specified decimals."""
    real, imag = c.as_real_imag()
    real = round(float(N(real)), decimals)
    imag = round(float(N(imag)), decimals)
    if imag >= 0:
        return f"{real} + {imag}*I"
    else:
        return f"{real} - {abs(imag)}*I"

def round_complex_list(solutions, decimals=5):
    """Round a list of complex SymPy solutions."""
    rounded = []
    for s in solutions:
        rounded.append(round_complex_number(s, decimals))
    return rounded

#----- Solve for x functions -----
def solve_for_x_rounded(expression, decimals=5):
    from sympy import symbols, Eq, solve, sympify
    x = symbols('x')
    
    if "=" not in expression:
        return None

    left, right = expression.split("=")
    left = sympify(left)
    right = sympify(right)
    equation = Eq(left, right)
    solutions = solve(equation, x)
    
    # Round solutions
    rounded_solutions = round_complex_list(solutions, decimals)
    
    return {"raw": solutions, "rounded": rounded_solutions}

def solve_for_x(expression, decimals=5, guess=1.0):
    x = symbols('x')

    # Split into left and right of '='
    if "=" not in expression:
        return None

    left, right = expression.split("=")
    print(left, right)

    # Convert strings into sympy expressions
    try:
        print("Trying to sympify with conversion...")
        left = sympify(clean_for_sympy(convert_to_sympy(left)))
        print("Left sympified successfully.")
        right = sympify(clean_for_sympy(convert_to_sympy(right)))
        print("Right sympified successfully.")
    except Exception:
        try:
            print("Converting to Python math and trying eval...", "Left:", convert_to_sympy(left), "Right:", convert_to_sympy(right))
            print("Sympify with conversion failed, trying direct...")
            left = sympify(left)
            print("Left sympified successfully without conversion.")
            right = sympify(right)
            print("Right sympified successfully without conversion.")
        except Exception as e:
            left = convert_to_sympy(left)
            right = convert_to_sympy(right)
            try:
                left = simplify(left)
            except Exception as e:
                print("Simplify failed for left side, trying fallback...", "Expression:", left)
                left = re.sub(r"pow", r"", left)
                left = re.sub(r",", r") ** (", left)
            try:                
                right = simplify(right)
            except Exception as e:
                # replace pow with a space and replace , with ) ** (
                print("Simplify failed for right side, trying fallback...", "Expression:", right)
                right = re.sub(r"pow", r"", right)
                right = re.sub(r",", r"**", right)
    equation = Eq(left, right)
    print("Final equation to solve:", equation)
    try:
        solutions = solve(equation, x)
        print("Raw solutions from solve():", solutions)
        if solutions:
            print("Exact solution:", solutions)
            return solutions
    except:
        print("Falling back to numeric solution (nsolve)...")
        try:
            numeric_sol = nsolve(equation, x, guess)
            solutions = [numeric_sol]
            print("Numeric solution:", solutions)
        except Exception as e:
            print("Could not solve numerically:", e)
            return {"error": "No solution found"}

    # Round solutions
    rounded_solutions = round_complex_list(solutions, decimals)
    
    # Also print rounded numeric versions in terminal if you want
    print("Rounded solutions:", rounded_solutions)
    
    return {"raw": solutions, "rounded": rounded_solutions}

#----- Quadratic solver -----
def solve_quadratic(expr):
    expr = expr.replace("=0", "").replace("= 0", "").strip()
    print("Quadratic expression to solve:", expr)
    expr = convert_to_sympy(expr)

    expr_sym = sp.sympify(expr)
    vars_in_expr = list(expr_sym.free_symbols)

    if len(vars_in_expr) != 1:
        raise ValueError("Quadratic solver requires exactly one variable")

    var = vars_in_expr[0]

    print("Solving for variable:", var)
    print("Final expression:", expr_sym)

    roots_dict = sp.roots(expr_sym, var)
    print("Roots dictionary:", roots_dict)
    solutions = list(roots_dict.keys())
    print("Solutions found:", solutions)

    return solutions

def handle_taylor_series(raw, expr):
    from sympy import symbols, sympify

    x = symbols('x')

    # Default values
    about = 0
    order = 5

    # Detect order
    order_match = re.search(r"order\s+(\d+)", raw)
    if order_match:
        order = int(order_match.group(1))

    # Detect center point
    about_match = re.search(r"about\s+([-\d\.]+)", raw)
    if about_match:
        about = sympify(about_match.group(1))

    # Remove English words
    expr = expr.replace("taylor series of", "")
    expr = expr.replace("maclaurin series of", "")
    expr = expr.replace("taylor expansion", "")
    expr = re.sub(r"about\s+[-\d\.]+", "", expr)
    expr = re.sub(r"to order\s+\d+", "", expr)
    expr = expr.strip()

    # Convert to sympy
    expr_sym = sympify(convert_to_sympy(expr))

    # Compute Taylor series
    series_expr = sp.series(expr_sym, x, about, order + 1)
    print("Taylor series result:", series_expr)

    return series_expr

def numeric_limit_estimate(expr_sym, var, point, max_abs=1e9):
    """Heuristic numeric probe for limits. Returns `sp.oo`, `-sp.oo`, a numeric value, or None.

    Strategy:
    - For finite approach points, evaluate the expression at points approaching from both sides
      (decreasing eps). If both sides grow in magnitude and signs are consistent, return +/-oo.
    - For infinite approach points, sample at increasing magnitudes and look for monotone growth.
    - This is heuristic and used as a last-resort when symbolic methods are inconclusive.
    """
    from sympy import lambdify
    import mpmath as mp

    try:
        f = lambdify(var, expr_sym, modules=["mpmath", "math"])
    except Exception:
        return None

    # Handle approach to infinity
    try:
        if point == sp.oo:
            xs = [10**k for k in range(1, 6)]
            vals = []
            for x in xs:
                try:
                    vals.append(mp.mpf(f(x)))
                except Exception:
                    vals.append(None)
            vals = [v for v in vals if v is not None]
            if not vals:
                return None
            abs_vals = [abs(v) for v in vals]
            if all(abs_vals[i] < abs_vals[i+1] for i in range(len(abs_vals)-1)):
                sgn = mp.sign(vals[-1])
                return sp.oo if sgn > 0 else -sp.oo
            return None

        if point == -sp.oo:
            xs = [-10**k for k in range(1, 6)]
            vals = []
            for x in xs:
                try:
                    vals.append(mp.mpf(f(x)))
                except Exception:
                    vals.append(None)
            vals = [v for v in vals if v is not None]
            if not vals:
                return None
            abs_vals = [abs(v) for v in vals]
            if all(abs_vals[i] < abs_vals[i+1] for i in range(len(abs_vals)-1)):
                sgn = mp.sign(vals[-1])
                return sp.oo if sgn > 0 else -sp.oo
            return None
    except Exception:
        pass

    # Finite approach point
    try:
        pt = float(N(point))
    except Exception:
        return None

    signs = []
    diverging = True
    for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        try:
            l = mp.mpf(f(pt - eps))
            r = mp.mpf(f(pt + eps))
        except Exception:
            # If evaluation raises (e.g., division by zero), treat as potential divergence and continue
            continue
        l_abs = abs(l)
        r_abs = abs(r)
        # consider it diverging if magnitudes exceed a large threshold consistently
        if l_abs > 1e6 and r_abs > 1e6 and mp.sign(l) == mp.sign(r):
            signs.append(mp.sign(l))
            continue
        else:
            diverging = False
            break

    if diverging and signs:
        if all(s > 0 for s in signs):
            return sp.oo
        if all(s < 0 for s in signs):
            return -sp.oo
    return None

def smart_limit(expr_sym, var, point):
    """Attempt to evaluate limit robustly. Returns a SymPy limit (including +/-oo) or None."""
    print("Entering series limit computation...")
    try:
        lim = sp.limit(expr_sym, var, point)
        # Accept any concrete symbolic result (including +/-oo) but ignore NaN
        if lim is not sp.nan:
            return lim
    except Exception:
        pass

    # Try Taylor series fallback
    for order in range(4, 12):
        try:
            series_expr = sp.series(expr_sym, var, point, order).removeO()
            simplified = sp.simplify(series_expr)
            lim = sp.limit(simplified, var, point)
            if lim is not sp.nan:
                return lim
        except Exception:
            continue

    # Last resort: numeric probe
    try:
        num_probe = numeric_limit_estimate(expr_sym, var, point)
        return num_probe
    except Exception:
        pass

    return None

#---------Checks for multi-equation systems---------
def contains_imaginary(equations):
    """Return True if any equation contains 'i' or 'I' (imaginary unit)."""
    for eq in equations:
        if re.search(r'\b(i|I)\b', eq):
            return True
    return False

def contains_trig(equations):
    """Return True if any equation contains trig functions."""
    trig_keywords = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'arcsin', 'arccos', 'arctan']
    for eq in equations:
        if any(k in eq for k in trig_keywords):
            return True
    return False

#----- Multi-equation system splitter and solver -----
def split_system(text):
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)
    # Split on "and", "next equation", or semicolons
    text = re.sub(r'\bnext equation\b', ' and ', text, flags=re.IGNORECASE)
    # safely spltit
    parts = re.split(r'\band\b|;+', text, flags=re.IGNORECASE)
    
    #Clean up
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
    
        p = re.sub(r'^(and|then|the|also)\b', '', p, flags=re.IGNORECASE).strip()
        p = re.sub(r'^\*+', '', p)  # remove leading *
        out.append(p)
        print("Split equation:", p)
    
    return out

def solve_complex_system_real(eqs_sym, syms_list, max_attempts=15):
    import numpy as np
    from scipy.optimize import least_squares
    print("Solving complex system with solve_complex_system_real function")
    n = len(syms_list)
    # Split variables into real and imaginary parts
    def residuals(vals):
        subs = {}
        for i, s in enumerate(syms_list):
            subs[s] = vals[i] + 1j*vals[i + n]
        res = []
        for eq in eqs_sym:
            diff = eq.lhs - eq.rhs
            re, im = diff.subs(subs).as_real_imag()
            res.append(float(re))
            res.append(float(im))
        return np.array(res)

    rng = np.random.default_rng()
    for attempt in range(max_attempts):
        guess = rng.uniform(-5,5, 2*n)  # first n real, next n imag
        sol = least_squares(residuals, guess)
        print("Attempt", attempt+1, "solution:", sol.x)
        if sol.success and np.allclose(residuals(sol.x), 0, atol=1e-6):
            solution = {}
            for i, s in enumerate(syms_list):
                solution[str(s)] = sol.x[i] + 1j*sol.x[i+n]
            return solution
    return None

def solve_mixed_system(eqs_sym, syms_list, max_attempts=15):
    import numpy as np
    from scipy.optimize import least_squares
    import sympy as sp
    
    print("Solving mixed real/complex/trig system")
    
    n = len(syms_list)
    
    print("Checking if system is complex...")
    # Detect whether system becomes complex
    def system_is_complex():
        for eq in eqs_sym:
            expr = eq.lhs - eq.rhs
            if expr.has(sp.I):
                print("System is complex due to equation:", eq)
                return True
        print("System is not complex.")
        return False
    
    
    
    complex_system = system_is_complex()
    
    def residuals(vals):
        subs = {}
        # Map symbols to current values
        if complex_system:
            # First n real parts, next n imaginary parts
            for i, s in enumerate(syms_list):
                subs[s] = vals[i] + 1j * vals[i + n]
        else:
            for i, s in enumerate(syms_list):
                subs[s] = vals[i]
        
        res = []
        # Evaluate each equation
        for eq in eqs_sym:
            expr = eq.lhs - eq.rhs
            val = expr.subs(subs)
            # Force numeric evaluation
            if complex_system:
                re, im = sp.N(val).as_real_imag()
                res.append(float(re))
                res.append(float(im))
            else:
                res.append(float(sp.N(val)))
        
        return np.array(res, dtype=float)
        
    rng = np.random.default_rng()
    
    dim = 2 * n if complex_system else n
    
    print("Starting attempts to solve mixed system...")
    for attempt in range(max_attempts):
        guess = rng.uniform(-5, 5, dim)
        
        sol = least_squares(residuals, guess)
        print("Attempt", attempt+1, "solution:", sol.x)
        # Check for success
        if sol.success and np.allclose(residuals(sol.x), 0, atol=1e-6):
            solution = {}
            if complex_system:
                for i, s in enumerate(syms_list):
                    solution[str(s)] = sol.x[i] + 1j * sol.x[i + n]
                print("Complex system solution found.")
                return solution
            else:
                for i, s in enumerate(syms_list):
                    solution[str(s)] = sol.x[i]
                print("Real system solution found.")
                return solution
    print("No valid solution found for mixed system.")
    return None

def solve_trig_system_real(eqs_sym, syms_list, max_attempts=15):
    import numpy as np
    from scipy.optimize import least_squares
    import sympy as sp

    print("Solving trig system with solve_trig_system_real function")

    n = len(syms_list)

    def residuals(vals):
        # Substitute real values directly
        subs = {syms_list[i]: vals[i] for i in range(n)}

        res = []
        for eq in eqs_sym:
            diff = eq.lhs - eq.rhs
            val = diff.subs(subs)

            # Force numeric evaluation (trig-safe)
            res.append(float(sp.N(val)))

        return np.array(res, dtype=float)

    rng = np.random.default_rng()

    for attempt in range(max_attempts):
        # Same guessing strategy style as complex solver
        guess = rng.uniform(-5, 5, n)

        sol = least_squares(residuals, guess)

        if sol.success and np.allclose(residuals(sol.x), 0, atol=1e-6):
            solution = {}
            for i, s in enumerate(syms_list):
                solution[str(s)] = sol.x[i]
            return solution

    return None

def solve_real_system(eqs_sym, syms_list, max_attempts=15):
    import numpy as np
    from scipy.optimize import least_squares

    print("Solving real system with solve_real_system function")

    n = len(syms_list)

    def residuals(vals):
        # Map symbols to current values
        subs = {s: vals[i] for i, s in enumerate(syms_list)}

        res = []
        for eq in eqs_sym:
            # Support both Eq(lhs, rhs) and plain expressions
            if hasattr(eq, "lhs"):
                diff = eq.lhs - eq.rhs
            else:
                diff = eq

            val = diff.subs(subs)

            # Ensure numeric float
            res.append(float(val))

        return np.array(res)

    rng = np.random.default_rng()

    for attempt in range(max_attempts):
        guess = rng.uniform(-5, 5, n)

        sol = least_squares(residuals, guess)
        print("Attempt", attempt+1, "solution:", sol.x)

        if sol.success and np.allclose(residuals(sol.x), 0, atol=1e-6):
            solution = {str(s): sol.x[i] for i, s in enumerate(syms_list)}
            return solution

    return None

# --- Calculate function ---
def calculate():
    global use_radians
    global substitution
    if (use_radians == True or use_radians == False):
        print("Calculating...")
        raw = entry.get()
        print("Raw input:", raw)
        if "limit" in raw.lower():
            raw = re.sub(
            r"(approaches\s+[a-zA-Z0-9\.\-]+)\s+",r"\1, ",raw,flags=re.IGNORECASE)
            print("Modified raw for limit:", raw)
        if "integral" in raw.lower() or "integrate" in raw.lower():
            # Look for explicit substitution phrases (e.g., 'let u = ...', 'substitute u = ...')
            if re.search(r"\b(?:let|substitute|using substitution|u-substitution|use substitution)\b", raw, flags=re.IGNORECASE):
                try:
                    print("Integral detected, parsing substitution if any")
                    sub = parse_substitution(raw)
                    if sub:
                        substitution = True
                        u_var, u_rhs = sub
                        print(f"Detected substitution: {u_var} = {u_rhs}")
                        raw = raw.replace(f"{u_var} = {u_rhs}", "").strip()
                        raw= re.sub(r"\b(?:let|substitute|using substitution|u-substitution|use substitution)\b", "", raw, flags=re.IGNORECASE).strip()
                        print("Raw after removing substitution phrase:", raw)
                        # Test conversion
                        u_var = symbols(u_var)
                        u_rhs = convert_to_sympy(words_to_math(u_rhs))
                        print("Substitution after conversion:", u_var, "=", u_rhs)
                except Exception as e:
                    print("Error parsing substitution:", e)
        expr = words_to_math(raw)
        print("Converted expression:", expr)
        print("Success Converting")
        # Quadratic
        if (("=0" in expr or "= 0" in expr) and "and" not in expr.lower() and "next equation" not in expr.lower()):
            print("Solving Quadratic")
            solutions = solve_quadratic(expr)

            pretty = []
            for s in solutions:
                s_num = sp.N(s, 12)
                pretty.append(str(s_num))

            output_label.config(
            text="Quadratic solutions:\n" + "\n".join(pretty)
            )
            return
        print("Not a Quadratic")
        # Simplifaction
        if "simplify" in raw.lower() or "simplified" in raw.lower() or "simplification" in raw.lower():
            print("Simplification requested")
            expr_simp = Simplify_expression(expr)
            output_label.config(text=f"Result: {expr_simp}")
            return
        print("Does not need simplifcation")
        # Derivitives
        if "derivative" in raw.lower() or "differentiate" in raw.lower():
            try:
                print("Derivative requested")

                # ---- detect order ----
                order_map = {"first":1, "second":2, "third":3, "fourth":4}
                order = 1
                for word, n in order_map.items():
                    if word in expr:
                        order = n
                        expr = expr.replace(word, "")
                        break
                # ---- remove derivative words ----
                expr = expr.replace("derivative of", "").replace("differentiate", "").strip()
                expr = expr.lstrip("*")
                print("Expression after cleaning:", expr)

                # ---- detect evaluation ----
                value = None
                if " at x =" in expr:    
                    expr, value = expr.split(" at x =", 1)
                    value = sympify(convert_to_sympy(value))
                print("Expression to differentiate:", expr)
                print("Order of derivative:", order)

                # ---- symbolic conversion ----
                # Balance parentheses (fix stray/missing closes) before sympifying
                expr = fix_unmatched_parens(expr)
                if DEBUG:
                    print("Expression after paren fix:", expr)
                expr_sym = sympify(convert_to_sympy(expr))
                x = symbols("x")
                print("Symbolic expression:", expr_sym)
                # ---- apply derivative ----
                for _ in range(order):
                    expr_sym = sp.diff(expr_sym, x)

                # ---- evaluate if needed ----
                if value is not None:
                    expr_sym = expr_sym.subs(x, value)
                    print("Evaluated derivative at x =", value, ":", expr_sym)

                # Simplify for readability and try trig simplifications
                try:
                    simplified = sp.simplify(expr_sym)
                    try:
                        simplified = sp.trigsimp(simplified)
                    except Exception:
                        pass
                    print("Derivative result:", simplified)
                except Exception as e:
                    print("Could not simplify derivative:", e)
                    simplified = expr_sym

                output_label.config(text=f"Derivative:\n{simplified}")
                print ("Derivative computed successfully.")
                return

            except Exception as e:
                output_label.config(text=f"Derivative error: {e}")
                print("Derivative error:", e)
                return
        print("Not a derivative")
        # Integrals
        if "integral" in raw.lower() or "integrate" in raw.lower():
            if " from " in expr and " to " in expr:
                try:
                    print("Definite integral requested")

                    # Remove integral words
                    expr = re.sub(r"(integral of|integrate|integral)", "", expr)
                    expr = expr.strip()

                    # Split into main expression and bounds
                    main, bounds = expr.split(" from ", 1)
                    lower, upper = bounds.split(" to ", 1)

                    # Convert spoken math → symbolic math
                    integrand = sympify(convert_to_sympy(main))
                    lower = sympify(convert_to_sympy(lower))
                    upper = sympify(convert_to_sympy(upper))

                    # Detect potential substitution in the spoken raw input
                    sub = parse_substitution(expr_str)
                    if sub:
                        u_var, u_rhs = sub
                        try:
                            u_sym = symbols(u_var)
                            u_expr_sym = sympify(convert_to_sympy(u_rhs))
                            var = list(integrand.free_symbols)[0]
                            du_dx = sp.diff(u_expr_sym, var)
                            q = sp.simplify(integrand / du_dx)
                            q_sub = q.subs(u_expr_sym, u_sym)
                            lower_u = sp.simplify(u_expr_sym.subs(var, lower))
                            upper_u = sp.simplify(u_expr_sym.subs(var, upper))

                            # If q_sub is purely in u, integrate over transformed bounds
                            if q_sub.free_symbols.issubset({u_sym}):
                                print("Performing definite integral via u-substitution")
                                result_u = sp.integrate(q_sub, (u_sym, lower_u, upper_u))
                                output_label.config(text=f"Definite Integral (u-subst):\n{sp.simplify(result_u)}")
                                return

                            # Try fallback: attempt direct integration in u space
                            try:
                                result_u = sp.integrate(q_sub, (u_sym, lower_u, upper_u))
                                output_label.config(text=f"Definite Integral (u-subst fallback):\n{sp.simplify(result_u)}")
                                return
                            except Exception as e:
                                print("Definite u-substitution fallback failed:", e)
                                # continue to standard integrate
                        except Exception as e:
                            print("Error parsing substitution for definite integral:", e)

                    var = list(integrand.free_symbols)[0]
                    result = sp.integrate(integrand, (var, lower, upper))

                    output_label.config(text=f"Definite Integral:\n{result}")
                    return
                except Exception as e:
                    output_label.config(text=f"Definite Integral error: {e}")
                    print("Definite Integral error:", e)
            elif substitution == True:
                try:
                    print("Symbolic integral requested")

                    expr_str = re.sub(r"(integral of|integrate|integral)", "", expr)
                    expr_str = expr_str.strip()
                    print("Integral after removing:", expr_str)
 
                    # Remove any trailing differential words like 'dx' or 'du' for parsing
                    expr_clean = re.sub(r"\bd\s*[a-zA-Z]\b", "", expr_str).strip() 
                    print("Expression after removing differential:", expr_clean)
                    # Fix unmatched parentheses before sympifying
                    expr_clean = fix_unmatched_parens(expr_clean)

                    try:
                        x = symbols('x')
                        print("Variable for integration:", x)
                        print(u_var, "=", u_rhs)
                        # If a substitution like 'let u = ...' was found, attempt u-substitution
                        if substitution==True:
                            print("Attempting u-substitution...")
                            u_sym = u_var
                            print(f"Preparing u-substitution with {u_var} = {u_rhs}")
                            u_expr_sym = sympify(convert_to_sympy(u_rhs))

                            integrand_sym = sympify(convert_to_sympy(expr_clean))

                            print("Integrand for u-substitution:", integrand_sym)

                            du_dx = sp.diff(u_expr_sym, x)
                            print("u_expr:", u_expr_sym, "du/dx:", du_dx)

                            # Standard approach: integrand / du_dx, then replace u_expr -> u and integrate wrt u
                            q = sp.simplify(integrand_sym / du_dx)
                            print("Transformed integrand q:", q)
                            q_sub = q.subs(u_expr_sym, u_sym)
                            print("Substituted integrand q_sub:", q_sub)

                            # If q_sub is purely in u, integrate directly
                            print("Free symbols in q_sub:", q_sub.free_symbols)
                            if q_sub.free_symbols.issubset({u_sym}):
                                print("Performing u-substitution (pure u) with integrand:", q_sub)
                                res_u = sp.integrate(q_sub, u_sym)
                                final = res_u.subs(u_sym, u_expr_sym)
                                final = sp.simplify(final)
                                output_label.config(text=f"Integral (u-subst):\n{final} + C")
                                return

                            # Otherwise, try integrating symbolically in u (SymPy may succeed even if mixed)
                            try:
                                print("Trying fallback integrate in u for:", q_sub)
                                res_u = sp.integrate(q_sub, u_sym)
                                final = res_u.subs(u_sym, u_expr_sym)
                                final = sp.simplify(final)
                                output_label.config(text=f"Integral (u-subst fallback):\n{final} + C")
                                print("u-substitution fallback succeeded.", final)
                                return
                            except Exception as e:
                                print("u-substitution fallback failed:", e)

                            # If user provided an integrand in terms of u (e.g., 'sin u du'), integrate directly and substitute back
                            if re.search(r"\bdu\b", raw.lower()):
                                try:
                                    integrand_no_du = re.sub(r"\bdu\b", "", expr_str).strip()
                                    integrand_u = sympify(convert_to_sympy(integrand_no_du))
                                    res_u = sp.integrate(integrand_u, u_sym)
                                    final = res_u.subs(u_sym, u_expr_sym)
                                    final = sp.simplify(final)
                                    output_label.config(text=f"Integral (u variable given):\n{final} + C")
                                    print("Integrated directly in u variable.", final)
                                    return
                                except Exception as e:
                                    print("Direct u integration failed:", e)

                            # If all substitution attempts fail, fall back to standard integration
                            print("u-substitution attempts failed; falling back to standard integrate")

                        # No substitution or substitution failed: do standard integral
                        expr_sym = sympify(convert_to_sympy(expr_clean))
                        print("Expression for integral:", expr_sym)
                        result = sp.integrate(expr_sym, x)

                        output_label.config(text=f"Integral:\n{result}")
                        return

                    except Exception as e:
                        output_label.config(text=f"Integral error: {e}")
                        return

                except Exception as e:
                    output_label.config(text=f"Integral error: {e}")
                    return
        print("Not an integral")
        # Multi-equation system
        if "next equation" in raw.lower() or "and" in raw.lower():
            print("Detected multi-equation system.")
            equations = split_system(expr)
            print("Equations detected:", equations)

            # Convert to sympy Eq objects
            eqs_sym = []
            syms_set = set()
            for eq in equations:
                if not eq.strip():
                    continue
                left, right = eq.split("=", 1)
                L = sympify(convert_to_sympy(left.strip()))
                R = sympify(convert_to_sympy(right.strip()))
                eqs_sym.append(Eq(L, R))
                syms_set |= (L.free_symbols | R.free_symbols)
                print("Parsed equation:", L, "=", R)


            syms_list = sorted(syms_set, key=lambda s: s.name)
            print("Symbols detected:", syms_list)

            # Solve in real space (split complex variables)
            has_imag = contains_imaginary(equations)
            has_trig = contains_trig(equations)
            print("Contains imaginary:", has_imag)
            print("Contains trig functions:", has_trig)

            if has_trig & has_imag:
                print("Both imaginary and trigonometric functions detected.")
                sol = solve_mixed_system(eqs_sym, syms_list)
            elif has_trig:
                print("Solving trigonometric system...")
                sol = solve_trig_system_real(eqs_sym, syms_list)
            elif has_imag:
                print("Solving complex system in real space...")
                sol = solve_complex_system_real(eqs_sym, syms_list)
            else:  # not has_imag and not has_trig
                print("Solving system numerically...")
                sol = solve_real_system(eqs_sym, syms_list)
                print("Real system solution:", sol)
                #if isinstance(sol_sympy, list) and len(sol_sympy) > 0:
                 #   sol_sympy = sol_sympy[0]  # take first solution
                sol = {str(k): v for k, v in sol.items()}
                output_label.config(text="No valid numeric solution found.")
            
            if sol is None:
                output_label.config(text="No valid numeric solution found.")
                print("No valid solution found.")
            else:
                print("System solution found:", sol)
                sol_str = ""
                for v, val in sol.items():
                    # Only show complex if the equation had imaginary numbers
                    if has_imag and isinstance(val, complex):
                        if abs(val.imag) < 1e-12:
                            v = to_python_number(val)
                            val_str = f"{val.real:.12g}"
                        else:
                            val_str = f"{val.real:.12g} + {val.imag:.12g}i"
                    else:
                        val_str = f"{N(val):.12g}"
                    sol_str += f"{v} = {val_str}\n"
                output_label.config(text=f"Solutions:\n{sol_str}")
                print("System solution:\n", sol_str)
            return
        print("Not a system of equation")
        #Limits
        if "limit" in raw.lower():
            try:
                print("Limit requested")

                # Remove limit keywords
                expr_str = re.sub(r"\b(limit|limit of|find the limit of|calculate the limit of)\b\s*","",expr,flags=re.I).strip()

                # Normalize whitespace
                expr_str = re.sub(r"\s+", " ", expr_str)
                print("Expression for limit (string):", expr_str)

                # Find "as x approaches value"
                match = re.search(r"([a-zA-Z])\s+approaches\s+([^,]+),\s*(.*)", expr_str.lower())

                if not match:
                    raise ValueError("Could not parse limit structure")
                var_name = match.group(1)
                print("Variable part:", var_name)
                approach_raw = match.group(2).strip()
                print("Approach value part:", approach_raw)
                expr_raw = match.group(3)
                print("Raw expression part:", expr_raw)

                # Everything before "as x approaches" is the expression
                main = expr_raw.strip()
                print("Main expression:", main)
                print("Variable:", var_name)
                print("Approach value (raw):", approach_raw)

                # Convert variable
                var = symbols(var_name)
                print("Symbolic variable:", var)

                # Convert expression (you handle word→math elsewhere)
                main_converted = convert_to_sympy(main)
                print("Converted main expression:", main_converted)
                main_converted = main_converted.replace("math.", "")
                print("Main expression without 'math.':", main_converted)
                main_sym = sympify(main_converted)
                print("Symbolic main expression:", main_sym)

                # Convert approach value
                approach_converted = convert_to_sympy(approach_raw)
                print("Converted approach value:", approach_converted)
                approach_converted = approach_converted.replace("math.", "")
                print("Approach value without 'math.':", approach_converted)
                approach_val = sympify(approach_converted)
                print("Symbolic approach value:", approach_val)

                print("Converted for limit:", main_sym, var, approach_val)

                # Try normal limit
                result = sp.limit(main_sym, var, approach_val)
                print("Computed limit:", result)

                # If sympy returned an unevaluated Limit or NaN, attempt smarter fallbacks
                if isinstance(result, sp.Limit) or result is sp.nan:
                    print("Unevaluated or NaN limit returned — trying series/numeric fallbacks")
                    series_result = smart_limit(main_sym, var, approach_val)
                    if series_result is not None:
                        output_label.config(text=f"Limit (via series):\n{series_result}")
                        return
                    # Final numeric probe as a last resort
                    num_probe = numeric_limit_estimate(main_sym, var, approach_val)
                    if num_probe is not None:
                        output_label.config(text=f"Limit (numeric):\n{num_probe}")
                        return
                    output_label.config(text="Limit: Indeterminate form (unsolved)")
                    return

                # Detect 0/0 indeterminate forms and try series expansion
                num, den = sp.fraction(main_sym)
                print("Numerator:", num)
                print("Denominator:", den)
                print("Num at approach:", num.subs(var, approach_val))
                print("Den at approach:", den.subs(var, approach_val))
                
                num_lim = sp.limit(num, var, approach_val)
                den_lim = sp.limit(den, var, approach_val)

                if num_lim == 0 and den_lim == 0:
                    print("Indeterminate form detected — attempting series expansion")
                    series_result = smart_limit(main_sym, var, approach_val)
                    if series_result is not None:
                        output_label.config(text=f"Limit (via series):\n{series_result}")
                        return
                    else:
                        # Try numeric probe before giving up
                        num_probe = numeric_limit_estimate(main_sym, var, approach_val)
                        if num_probe is not None:
                            output_label.config(text=f"Limit (numeric):\n{num_probe}")
                            return
                        output_label.config(text="Limit: Indeterminate form (unsolved)")
                        return

                # Valid result (could be +/-oo) — show it
                output_label.config(text=f"Limit:\n{result}")
                return

            except Exception as e:
                output_label.config(text=f"Limit error: {e}")
                print("Limit error:", e)
                return
        print("No limit reqeuested")
        # Single equation
        if ("=" in expr or any(v in expr for v in ["x", "y", "z"]) or ("equals" in raw.lower())):
            print("Solve for X requested")
            sol = solve_for_x(expr)
            print("Raw solution:", sol)
            if sol is None:
                output_label.config(text="No equation to solve.")
                return
            if isinstance(sol, dict) and "rounded" in sol:
                output_label.config(text=f"Solutions:\n{sol['rounded']}")
            else:
                output_label.config(text="No solution found.")
                return
        print("Not solving for x")
        if (("=" in expr)):
            print("= is in the equation")
        if ((any(v in expr for v in ["x", "y", "z"]))):
            print("Varibles are detected")
        if (("equals" in raw.lower())):
            print("Word equals is detected")
        #Taylor Series
        if "taylor series" in raw.lower() or "taylor expansion" in raw.lower() or "maclaurin series" in raw.lower():
            try:
                print("Taylor series requested")
                result = handle_taylor_series(raw.lower(), expr)
                output_label.config(text=f"Taylor Series:\n{result}")
                return
            except Exception as e:
                output_label.config(text=f"Taylor series error: {e}")
                return
        print("Not a taylor series")
        # Plain calculation
        try:
            print("Plain calculation...")
            result = (sympify(clean_for_sympy(convert_to_sympy(expr))))
            print("Sympify:", result)
            result = N(result)
            print("After N: ", result)
            if not use_radians:
                result = result * math.pi / 180
            output_label.config(text=f"Result: {result}")
        except Exception as e:
            output_label.config(text=f"Error evaluating: {e}")
            print("Error evaluating expression:", e)
    else:
        print("Invalid state for use_radians:", use_radians)

# --- GUI setup ---
root = tk.Tk()
root.title("Whisper Voice Calculator")
root.geometry("350x400")

title = tk.Label(root, text="Whisper Voice Calculator", font=("Arial", 14))
title.pack(pady=10)

entry = tk.Entry(root, width=25)
entry.pack(pady=5)

calc_button = tk.Button(root, text="Calculate", command=button_cal)
calc_button.pack(pady=5)

start_button = tk.Button(root, text="🎤 Start Recording", command=start_recording)
start_button.pack(pady=5)

stop_button = tk.Button(root, text="🛑 Stop Recording", command=stop_recording, state="disabled")
stop_button.pack(pady=5)

output_label = tk.Label(root, text="Result: ", font=("Arial", 12))
output_label.pack(pady=10)

rad_checkbox = tk.Checkbutton(root, text="Use Radians", command=toggle_radians)
rad_checkbox.pack()
rad_checkbox.select()  # default checked

root.mainloop()
