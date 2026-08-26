import re

from num2words import num2words


class LiteNormalizer:
    """Drop-in replacement for NeMo Normalizer (pynini does not build on Apple Silicon).

    Handles the cases that matter for alignment transcripts coming from ASR:
    digits -> words, common currency/percent patterns, ordinal suffixes.
    The downstream `clean()` in FACodec_AC.utils handles punctuation stripping.
    """

    _ORDINALS = {
        "1st": "first", "2nd": "second", "3rd": "third",
        "4th": "fourth", "5th": "fifth", "6th": "sixth",
        "7th": "seventh", "8th": "eighth", "9th": "ninth",
        "10th": "tenth",
    }

    def __init__(self, lang: str = "en"):
        self.lang = lang

    def normalize(self, txt: str) -> str:
        txt = txt.replace("%", " percent")
        txt = re.sub(r"\$\s*(\d+(?:\.\d+)?)", lambda m: self._money(m.group(1)), txt)
        txt = re.sub(r"(\d+(?:\.\d+)?)\s*percent", r"\1 percent", txt)
        for k, v in self._ORDINALS.items():
            txt = re.sub(rf"\b{re.escape(k)}\b", v, txt, flags=re.IGNORECASE)

        def num_repl(m):
            token = m.group(0)
            try:
                if "." in token:
                    return num2words(float(token))
                return num2words(int(token))
            except (ValueError, NotImplementedError):
                return token

        txt = re.sub(r"\d+(?:\.\d+)?", num_repl, txt)
        return txt

    @staticmethod
    def _money(amount: str) -> str:
        try:
            value = float(amount)
            dollars = int(value)
            cents = round((value - dollars) * 100)
            parts = [num2words(dollars), "dollars"]
            if cents:
                parts += ["and", num2words(cents), "cents"]
            return " ".join(parts)
        except (ValueError, NotImplementedError):
            return amount
