"""Compare numeric meaning, not Korean suffixes attached to a number."""
from decimal import Decimal
import re
import unicodedata

SCALE = {"억": Decimal(100000000), "만": Decimal(10000), "천": Decimal(1000),
         "백": Decimal(100), "십": Decimal(10), "": Decimal(1)}
AMOUNT = r"[+-]?(?:\d[\d,]*(?:\.\d+)?\s*[억만천백십]?\s*)+"
MONEY = re.compile(r"(?<![\d.])(" + AMOUNT + r")원")
PERCENT = re.compile(r"(?<![\d.])([+-]?\d[\d,]*(?:\.\d+)?)\s*(?:%|퍼센트)")
OFFER = re.compile(r"(?<!\d)(\d+)\s*\+\s*(\d+)(?!\d)")
UNITS = "개월|세트|퍼센트|가지|kg|ml|mL|cm|mm|년|월|일|시|분|초|개|명|잔|팩|위|등|배|회|g|L|l"
NUMBER = re.compile(r"(?<![\d.])([+-]?\d[\d,]*(?:\.\d+)?)\s*(" + UNITS + r")?")


def decimal(value):
    return Decimal(value.replace(",", "").strip())


def won(value):
    value = value.replace(",", "").replace(" ", "")
    sign = -1 if value.startswith("-") else 1
    parts = re.findall(r"(\d+(?:\.\d+)?)([억만천백십]?)", value)
    return sign * sum((Decimal(number)*SCALE[scale] for number, scale in parts), Decimal(0))


def claims(text):
    text = unicodedata.normalize("NFKC", text)
    remaining = list(text)
    result = set()
    # Consume complete offers before interpreting their numbers as independent claims.
    for pattern, kind in ((OFFER, "offer"), (MONEY, "won"), (PERCENT, "percent")):
        for match in pattern.finditer("".join(remaining)):
            if kind == "offer":
                value = (int(match[1]), int(match[2]))
            else:
                value = won(match[1]) if kind == "won" else decimal(match[1])
            result.add((kind, value))
            remaining[match.start():match.end()] = " " * (match.end()-match.start())
    for match in NUMBER.finditer("".join(remaining)):
        result.add(((match[2] or "number").lower(), decimal(match[1])))
    return result


def price_context(brief):
    result = {"sale_price_won": brief.price, "original_price_won": brief.original_price}
    if brief.price is not None and brief.original_price is not None:
        saving = brief.original_price-brief.price
        exact = Decimal(saving)*100/Decimal(brief.original_price)
        result.update(saving_won=saving, discount_percent_display=int(exact),
                      discount_percent_exact=str(exact))
    return result


def allowed_claims(brief):
    # Internal IDs, style instructions and technical settings are not ad facts.
    source = "\n".join([brief.product, brief.message, brief.period, brief.contact,
                       brief.store.name, brief.store.region, brief.store.address])
    result = claims(source)
    for amount in (brief.price, brief.original_price):
        if amount is not None:
            result.update({("won", Decimal(amount)), ("number", Decimal(amount))})
    price = price_context(brief)
    if "saving_won" in price:
        result.add(("won", Decimal(price["saving_won"])))
        result.add(("percent", Decimal(price["discount_percent_display"])))
        # Permit exact finite discounts (12.5%), never a rounded-up promise.
        exact = Decimal(price["discount_percent_exact"])
        if exact.as_tuple().exponent >= -2:
            result.add(("percent", exact))
    return result


def unsupported_claims(text, brief):
    return claims(text) - allowed_claims(brief)
