"""arff.py — ARFF loader + scaler, verbatim ssffs.cpp:99-330"""

from __future__ import annotations

import numpy as np


class ArffData:
    def __init__(self, n_features: int, n_classes: int, class_size: list[int], data: np.ndarray):
        self.n_features = n_features
        self.n_classes = n_classes
        self.class_size = class_size  # per class
        self.data = data  # flat double, class-major [class][sample][feature]

    def total_samples(self) -> int:
        return int(sum(self.class_size))


def _die(msg: str):
    raise ValueError(f"ssffs: {msg}")


def _upper(s: str) -> str:
    return s.upper()


def _arff_trim(s: str) -> str:
    return s.strip(" \t\r\n")


def load_arff(path: str) -> ArffData:
    # read all lines
    try:
        with open(path, encoding="utf-8", errors="strict") as f:
            raw = f.read().splitlines()
    except FileNotFoundError:
        _die(f"cannot open '{path}'")
    attrs: list[dict] = []  # {name, nominal, is_class, values}
    class_attr = -1
    reading_data = False
    records: list[tuple[list[float], int]] = []  # (feat list[float], cls)
    n_features = 0
    n_classes = 0
    feat_of_attr: list[int] = []

    def prepare_attributes():
        nonlocal class_attr, n_features, n_classes, feat_of_attr
        for i, a in enumerate(attrs):
            un = _upper(a["name"])
            if (un in {"CLASS", "'CLASS'"}) and a["nominal"]:
                a["is_class"] = True
                class_attr = i
        feat_of_attr = [-1] * len(attrs)
        nf = 0
        for i, a in enumerate(attrs):
            if not a["is_class"]:
                feat_of_attr[i] = nf
                nf += 1
        if class_attr < 0:
            _die("no nominal 'class' attribute found")
        n_features = nf
        n_classes = len(attrs[class_attr]["values"])

    # helper to parse value token
    def read_attr_value(ai: int, feat: list[float], cls_holder: list[int], token: str):
        a = attrs[ai]
        fi = feat_of_attr[ai]
        if not a["nominal"]:
            try:
                # ssffs.cpp uses sscanf "%f" -> float then widen
                v = np.float32(float(token))
            except ValueError:
                _die(f"bad numeric value '{token}' for attribute {a['name']}")
            feat[fi] = float(v)  # widen to double via Python float
        else:
            val = -1
            for j, vv in enumerate(a["values"]):
                if vv == token:
                    val = j
                    break
            if val == -1:
                _die(f"nominal value '{token}' not found for attribute {a['name']}")
            if a["is_class"]:
                cls_holder[0] = val
            else:
                _die(f"nominal feature attributes are not supported ('{a['name']}')")

    # main parse loop
    for line in raw:
        # keep original for attribute parsing, but trim for control
        t = _arff_trim(line)
        if not t:
            continue
        if t.startswith("%"):
            continue
        if reading_data:
            if not t:
                continue
            rec_feat = [0.0] * n_features
            cls_holder = [-1]
            if t.startswith("{"):
                if not t.endswith("}"):
                    _die("unterminated sparse data row")
                inner = t[1:-1].strip()
                if inner:
                    # split by commas not inside quotes
                    parts = []
                    cur = ""
                    in_q = None
                    for ch in inner:
                        if in_q:
                            cur += ch
                            if ch == in_q:
                                in_q = None
                        elif ch in ("'", '"'):
                            in_q = ch
                            cur += ch
                        elif ch == ",":
                            parts.append(cur.strip())
                            cur = ""
                        else:
                            cur += ch
                    if cur.strip():
                        parts.append(cur.strip())
                    for part in parts:
                        if not part:
                            continue
                        # part is "id value" or "id 'value'"
                        # split into id and val by whitespace, respecting quoted val
                        part = part.strip()
                        # find first whitespace
                        p = 0
                        while p < len(part) and part[p] not in (" ", "\t"):
                            p += 1
                        if p == len(part):
                            _die("malformed sparse data row")
                        id_str = part[:p].strip()
                        val_str = part[p:].strip()
                        # strip quotes from val_str if quoted
                        if val_str and val_str[0] in ("'", '"'):
                            qc = val_str[0]
                            if len(val_str) >= 2 and val_str[-1] == qc:
                                val_str = val_str[1:-1]
                            else:
                                # find closing quote
                                end = val_str.find(qc, 1)
                                if end != -1:
                                    val_str = val_str[1:end]
                        else:
                            val_str = _arff_trim(val_str)
                        try:
                            att_id = int(id_str)
                        except ValueError:
                            _die("sparse row: invalid attribute index")
                        if att_id < 0 or att_id >= len(attrs):
                            _die("sparse row: invalid attribute index")
                        read_attr_value(att_id, rec_feat, cls_holder, val_str)
                if cls_holder[0] == -1:
                    cls_holder[0] = 0
            else:
                # dense — need to split by commas respecting quotes
                tokens: list[str] = []
                cur = ""
                in_q = None
                for ch in t:
                    if in_q:
                        cur += ch
                        if ch == in_q:
                            in_q = None
                    elif ch in ("'", '"'):
                        in_q = ch
                        cur += ch
                    elif ch == ",":
                        tokens.append(cur)
                        cur = ""
                    else:
                        cur += ch
                tokens.append(cur)
                if len(tokens) != len(attrs):
                    _die("dense row ended too soon")
                for ci, tok in enumerate(tokens):
                    tok = _arff_trim(tok)
                    # strip outer quotes for nominal? keep as-is for read_attr_value comparison (values stored without quotes? stored trimmed)
                    # In ssffs.cpp, read_attr_value receives trimmed token including possibly quotes? Actually attribute values stored trimmed without quotes after parsing { }. For dense, token may be "'cls0'"? Need to handle quotes.
                    # If token is quoted, strip quotes for nominal lookup
                    if tok and tok[0] in ("'", '"') and len(tok) >= 2 and tok[-1] == tok[0]:
                        # keep inner for nominal? The values list stores as trimmed without quotes (stored via arff_trim). So strip.
                        inner_tok = tok[1:-1]
                        # For numeric, quoted numeric shouldn't happen; keep inner?
                        # Use inner for nominal class
                        # Detect if attribute is nominal -> use inner
                        if attrs[ci]["nominal"]:
                            tok = inner_tok
                    read_attr_value(ci, rec_feat, cls_holder, tok)
                if cls_holder[0] < 0:
                    _die("dense row without class value")
            records.append((rec_feat, cls_holder[0]))
        elif t.startswith("@"):
            # find key
            # ssffs.cpp: char *val = line+2; while (*val != ' ' && ...) val++; *val=0
            # then upper(line) for key
            # We replicate simpler
            upper_line = _upper(t)
            if upper_line.startswith("@RELATION"):
                continue
            if upper_line.startswith("@DATA"):
                reading_data = True
                prepare_attributes()
                continue
            if upper_line.startswith("@ATTRIBUTE"):
                # parse: @ATTRIBUTE <name> <type>
                # name may be quoted or single word, type is rest
                rest = t[len("@ATTRIBUTE") :].strip()
                if not rest:
                    _die("empty attribute definition")
                # name is first token (quoted or unquoted)
                if rest[0] in ("'", '"'):
                    qc = rest[0]
                    end = rest.find(qc, 1)
                    if end == -1:
                        _die("empty attribute definition")
                    name = rest[: end + 1]
                    type_str = rest[end + 1 :].strip()
                else:
                    # split by whitespace
                    parts = rest.split(None, 1)
                    name = parts[0]
                    type_str = parts[1] if len(parts) > 1 else ""
                type_str = _arff_trim(type_str)
                a = {"name": name, "nominal": False, "is_class": False, "values": []}
                if type_str.startswith("{"):
                    a["nominal"] = True
                    if not type_str.endswith("}"):
                        _die(f"malformed nominal attribute '{name}'")
                    inner = type_str[1:-1]
                    # split by commas
                    vals = []
                    cur = ""
                    in_q = None
                    for ch in inner:
                        if in_q:
                            cur += ch
                            if ch == in_q:
                                in_q = None
                        elif ch in ("'", '"'):
                            in_q = ch
                            cur += ch
                        elif ch == ",":
                            vals.append(_arff_trim(cur))
                            cur = ""
                        else:
                            cur += ch
                    vals.append(_arff_trim(cur))
                    # strip empty?
                    cleaned = []
                    for v in vals:
                        if v:
                            # remove outer quotes if present? Keep as stored trimmed (without outer quotes stripped? In C++ they store string as trimmed token including quotes? Actually they push std::string(arff_trim(t.data())) where t is from split by comma, so quotes remain inside string. But later comparison a.values[j]==value does direct string compare, so nominal value token must match exactly including quotes? However they also handle quoted values in data rows by stripping? Let's store as trimmed raw, but read_attr_value compares value string directly (passed trimmed token). For nominal data row, value may be without quotes. This could mismatch if attribute values were quoted. Safer to store trimmed without outer quotes similar to data token handling.
                            # If value is quoted, strip quotes
                            if len(v) >= 2 and v[0] in ("'", '"') and v[-1] == v[0]:
                                v = v[1:-1]
                            cleaned.append(v)
                    a["values"] = cleaned
                else:
                    tn = _upper(type_str)
                    if tn not in ("REAL", "NUMERIC", "INTEGER"):
                        _die(
                            f"unsupported attribute type '{type_str}' ('{name}') - this standalone loads numeric features and one nominal class only"
                        )
                attrs.append(a)
        else:
            _die(f"unexpected line before @DATA: '{t}'")
    if not reading_data:
        _die("no @DATA section found")
    if not records:
        _die("no data rows found")
    # class-major stable sort then widen to double
    class_size = [0] * n_classes
    for _, cls in records:
        if cls < 0 or cls >= n_classes:
            _die("record with invalid class id")
        class_size[cls] += 1
    total = len(records)
    flat: np.ndarray = np.zeros(total * n_features, dtype=np.float64)
    idx = 0
    for c in range(n_classes):
        for feat, cls in records:
            if cls == c:
                for ff in range(n_features):
                    flat[idx] = float(feat[ff])  # already double
                    idx += 1
    return ArffData(n_features, n_classes, class_size, flat)


def scale_to01(d: ArffData) -> None:
    samples = sum(d.class_size)
    n = d.n_features
    flat: np.ndarray = d.data
    for f in range(n):
        mn = mx = 0.0
        first = True
        # stride access: idx = f + p*n
        for p in range(samples):
            idx = f + p * n
            v = float(flat[idx])
            if first:
                mn = mx = v
                first = False
            else:
                mx = max(mx, v)
                mn = min(mn, v)
        if mx > mn:
            for p in range(samples):
                idx = f + p * n
                flat[idx] = (flat[idx] - mn) / (mx - mn)
        else:
            for p in range(samples):
                idx = f + p * n
                flat[idx] = 0.0
