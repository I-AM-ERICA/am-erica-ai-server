"""Measured word wrapping, atomic prices, and balanced short headings."""
from functools import lru_cache
import math
import re


def wrap_text(text, font, width, *, balanced=False):
    result = []
    for paragraph in text.split("\n"):
        # A price and its unit belong together, even when the user adds a space.
        paragraph = re.sub(r"(\d)\s+(원|만원|천원|%|퍼센트)(?=\s|$)", lambda m: m[1]+"\u00a0"+m[2], paragraph)
        tokens = re.findall(r"[^\s]+(?:\u00a0[^\s]+)*", paragraph.replace("\u00a0", "\ue000"))
        tokens = [token.replace("\ue000", "\u00a0") for token in tokens]
        words = []
        for token in tokens:
            if font.getlength(token) <= width or re.fullmatch(r"[\d,.]+\s*(?:원|만원|천원|%|퍼센트)?", token):
                words.append(token)
                continue
            # Only break a single overlong word at a syllable boundary.
            chunk = ""
            for char in token:
                if chunk and font.getlength(chunk+char) > width:
                    if char in ",.!?，。！？)]}”:;" and len(chunk) > 1:
                        words.append(chunk[:-1]); chunk = chunk[-1]+char
                    else:
                        words.append(chunk); chunk = char
                else:
                    chunk += char
            if chunk:
                words.append(chunk)
        if not words:
            result.append(""); continue
        lines, line = [], ""
        for word in words:
            candidate = line + " " + word if line else word
            if line and font.getlength(candidate) > width:
                lines.append(line); line = word
            else:
                line = candidate
        lines.append(line)
        if balanced and 1 < len(lines) <= 4 and len(words) <= 40:
            target = font.getlength(" ".join(words))/len(lines)
            @lru_cache(None)
            def best(start, count):
                if count == 0:
                    return (0, ()) if start == len(words) else (math.inf, ())
                optimum = (math.inf, ())
                for end in range(start+1, len(words)+1):
                    candidate = " ".join(words[start:end])
                    length = font.getlength(candidate)
                    if length > width:
                        break
                    score, rest = best(end, count-1)
                    score += (length-target)**2
                    if score < optimum[0]:
                        optimum = score, (candidate,)+rest
                return optimum
            _, chosen = best(0, len(lines))
            if chosen:
                lines = list(chosen)
        result.extend(lines)
    return result
