#!/usr/bin/env python3
"""
Shopping Sessions: columnar restructure of pipeline.py (same rules, same
outputs; pipeline.py stays the proven fallback).

Reads  $DATA_DIR    (default /data)
Writes $OUTPUT_DIR  (default /output): user_daily/ and campaign_daily/ as Parquet
Temp   $SCRATCH_DIR (default /scratch)

Why this exists: pipeline.py holds Python objects per event row through
parse/dedup/validate (about 0.6 GB per 1M events), which extrapolates past
the scorer's 24 GB at the 60M-event final. Here the row work lives in
Polars columns end to end, and every wide intermediate is either streamed
or avoided:

- Raw parse: files are read in ~32 MB byte blocks (the only Python loop is
  over blocks; splitting/decoding is C). The spec's CSV dialect is enforced
  with vectorized string expressions: comma count for the field count, a
  quote is legal only as exactly one pair wrapping a whole field (quote
  count 2 + first/last char), wrapped fields are unwrapped by slicing.
  Valid rows land in Parquet parts under SCRATCH with a global row number
  (_ord, in read order), so parse memory is bounded by the block size and
  the largest single block frame, never by the dataset or file size.
- Dedup: a narrow keys frame (event_id, ingest_ms, _ord) is sorted with
  the streaming engine and keep-first picks the winning copy; the full
  rows are then fetched with a semi join inside the same streaming plan
  that validates them, so the wide raw frame is never materialized.
- Timestamps: F1/F2/F3 conversion is arithmetic on string slices inside
  Polars (days-from-civil, leap-year month lengths, offset subtraction),
  gated by full-shape ASCII regexes and range checks, so it agrees with
  the strict Python parser in pipeline.py.
- Validation is filter expressions; person/category/campaign mappings are
  Polars joins; the collected events frame is already slim (5 columns,
  person and event_type as categoricals for the sessionize sort).
- Sessionization and credit (join_asof on per-timestamp collapsed clicks)
  are the same vectorized code as pipeline.py. Assists use pipeline.py's
  merged click-window intervals + range join, but chunked by person hash
  (~400k sessions per chunk, aggregated per chunk): the range join's
  transient is several times its ~100M-row output at final scale, and
  chunking is what bounds it.

What stayed Python, deliberately:
- The manifest window (two timestamps, parsed with the strict F1 parser).
- Union-find over links: link rows are far fewer than events, and the
  resulting ID -> person map is applied as a Polars join, never row loops.
"""
import glob
import json
import os
import shutil
import tempfile

NCPU = os.environ.get("NCPU")
if NCPU:
    # Must be set before polars is imported.
    os.environ.setdefault("POLARS_MAX_THREADS", NCPU)
_SCRATCH_ENV = os.environ.get("SCRATCH_DIR")
if _SCRATCH_ENV:
    os.environ.setdefault("POLARS_TEMP_DIR", _SCRATCH_ENV)

import polars as pl  # noqa: E402

DATA = os.environ.get("DATA_DIR", "/data")
OUT = os.environ.get("OUTPUT_DIR", "/output")

DAY_MS = 86_400_000
WEEK_MS = 604_800_000
SESSION_GAP_MS = 1_800_000


def _resolve_scratch():
    cand = os.environ.get("SCRATCH_DIR")
    if cand:
        try:
            os.makedirs(cand, exist_ok=True)
            if os.access(cand, os.W_OK):
                return cand
        except OSError:
            pass
    return tempfile.mkdtemp(prefix="shopping_sessions_")


SCRATCH = _resolve_scratch()

# --------------------------------------------------------------------------
# Manifest window: strict F1 parse (the rules say ingest_ts and the
# manifest are always valid F1; pipeline.py parses them the same way).
# --------------------------------------------------------------------------


def _ascii_digits(s):
    return s.isascii() and s.isdigit()


_DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _civil_ms(year, month, day, hour, minute, second):
    if year < 1 or month < 1 or month > 12 or hour > 23 or minute > 59 or second > 59:
        return None
    dim = _DAYS_IN_MONTH[month - 1]
    if month == 2 and year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        dim = 29
    if day < 1 or day > dim:
        return None
    y = year - 1 if month <= 2 else year
    era = y // 400
    yoe = y - era * 400
    doy = (153 * ((month + 9) % 12) + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    days = era * 146097 + doe - 719468
    return ((days * 24 + hour) * 60 + minute) * 60000 + second * 1000


def _parse_f1_py(value):
    if len(value) != 20 or value[19] != "Z":
        return None
    if (
        value[4] != "-"
        or value[7] != "-"
        or value[10] != "T"
        or value[13] != ":"
        or value[16] != ":"
    ):
        return None
    digits = (
        value[0:4] + value[5:7] + value[8:10]
        + value[11:13] + value[14:16] + value[17:19]
    )
    if not _ascii_digits(digits):
        return None
    return _civil_ms(
        int(value[0:4]), int(value[5:7]), int(value[8:10]),
        int(value[11:13]), int(value[14:16]), int(value[17:19]),
    )


with open(os.path.join(DATA, "manifest.json")) as f:
    manifest = json.load(f)
window_start = _parse_f1_py(manifest["window_start"])
window_end = _parse_f1_py(manifest["window_end"])
print("window:", manifest["window_start"], "to", manifest["window_end"])
print("polars threads:", pl.thread_pool_size())

# --------------------------------------------------------------------------
# Raw CSV parse, columnar. Files are read in byte blocks; each block's
# lines become a Polars frame where the dialect is checked with string
# expressions (see the module docstring). An empty field stays "" and is
# treated as null by every downstream check, exactly as pipeline.py does
# in its fused loops. Nothing is normalized: spaces and case survive.
# --------------------------------------------------------------------------

_BLOCK_BYTES = 1 << 25


def _line_blocks(path):
    """Yield lists of decoded lines (split on b"\\n" only, so a \\r stays an
    ordinary trailing character of the last field). The final yield is the
    tail after the last newline, or [""] when the file ends with one; an
    empty line is malformed and dropped downstream either way."""
    with open(path, "rb") as handle:
        leftover = b""
        while True:
            chunk = handle.read(_BLOCK_BYTES)
            if not chunk:
                break
            parts = (leftover + chunk).split(b"\n")
            leftover = parts.pop()
            if parts:
                yield [part.decode("utf-8") for part in parts]
        yield [leftover.decode("utf-8")]


def _lines_frame(lines, nfields):
    """Structurally valid rows of one block, as string columns c0..c{n-1}."""
    schema = {f"c{i}": pl.String for i in range(nfields)}
    if not lines:
        return pl.DataFrame(schema=schema)
    df = pl.DataFrame({"line": pl.Series("line", lines, dtype=pl.String)})
    # Wrong field count (an empty line included) is malformed.
    df = df.filter(pl.col("line").str.count_matches(",") == nfields - 1)
    if df.height == 0:
        return pl.DataFrame(schema=schema)
    df = df.with_columns(_fields=pl.col("line").str.split(","))
    exprs = []
    valid = pl.lit(True)
    for i in range(nfields):
        field = pl.col("_fields").list.get(i, null_on_oob=True)
        quotes = field.str.count_matches('"')
        wrapped = (quotes == 2) & field.str.starts_with('"') & field.str.ends_with('"')
        valid = valid & ((quotes == 0) | wrapped)
        # Unwrap a wrapped field by slicing off its two quotes; the length
        # is clipped at 0 so unwrapped fields (discarded by the when) can
        # never slice out of range.
        inner_len = (field.str.len_chars() - 2).clip(lower_bound=0)
        exprs.append(
            pl.when(wrapped)
            .then(field.str.slice(1, inner_len))
            .otherwise(field)
            .alias(f"c{i}")
        )
    return df.with_columns(exprs).filter(valid).select(list(schema))


def _iter_file_frames(path, nfields):
    """Yield the structurally valid rows of one file, block by block
    (first line is the header). A generator, so callers never hold more
    than one block frame per file."""
    header_pending = True
    for lines in _line_blocks(path):
        if header_pending:
            lines = lines[1:]
            header_pending = False
        if lines:
            yield _lines_frame(lines, nfields)


def _parse_to_frame(paths, nfields):
    schema = {f"c{i}": pl.String for i in range(nfields)}
    frames = []
    for path in paths:
        frames.extend(frame for frame in _iter_file_frames(path, nfields) if frame.height)
    if not frames:
        return pl.DataFrame(schema=schema)
    return pl.concat(frames, how="vertical")


_PARTS_DIR = os.path.join(SCRATCH, "raw_parts")


def _parse_to_parts(paths, nfields, tag):
    """Parse files into Parquet parts under SCRATCH, stamping each row
    with its global read-order number (_ord). Returns the part paths."""
    parts_dir = os.path.join(_PARTS_DIR, tag)
    shutil.rmtree(parts_dir, ignore_errors=True)
    os.makedirs(parts_dir, exist_ok=True)
    parts = []
    seq = 0
    offset = 0
    for path in paths:
        for frame in _iter_file_frames(path, nfields):
            if frame.height:
                frame = frame.with_row_index("_ord", offset=offset)
                offset += frame.height
                part = os.path.join(parts_dir, f"part-{seq:05d}.parquet")
                seq += 1
                frame.write_parquet(part)
                parts.append(part)
    return parts


def _scan_parts(parts, nfields):
    schema = {f"c{i}": pl.String for i in range(nfields)}
    schema["_ord"] = pl.UInt32
    if parts:
        return pl.scan_parquet(parts)
    return pl.DataFrame(schema=schema).lazy()


# --------------------------------------------------------------------------
# Reference data: products and campaigns (clean per the rules).
# --------------------------------------------------------------------------

products_frame = _parse_to_frame([os.path.join(DATA, "products.csv")], 2).rename(
    {"c0": "product_id", "c1": "category"}
)
products_frame = products_frame.filter(pl.col("product_id") != "")
campaigns_frame = _parse_to_frame([os.path.join(DATA, "campaigns.csv")], 2).rename(
    {"c0": "campaign_id", "c1": "channel"}
)
campaigns_frame = campaigns_frame.filter(pl.col("campaign_id") != "")

# --------------------------------------------------------------------------
# Linked IDs: union-find over links/ in Python (link rows are far fewer
# than events); the ID -> person map is applied later as a Polars join.
# A person's name is the byte-wise smallest ID in their component (Python
# str order matches UTF-8 byte order).
# --------------------------------------------------------------------------

parent = {}


def _find(x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


link_files = sorted(glob.glob(os.path.join(DATA, "links", "*.csv")))
links = _parse_to_frame(link_files, 2)
links = links.filter((pl.col("c0") != "") & (pl.col("c1") != ""))
for a, b in links.iter_rows():
    for x in (a, b):
        if x not in parent:
            parent[x] = x
    ra, rb = _find(a), _find(b)
    if ra != rb:
        parent[ra] = rb
del links

_smallest = {}
for member in parent:
    root = _find(member)
    if root not in _smallest or member < _smallest[root]:
        _smallest[root] = member
person_of = {member: _smallest[_find(member)] for member in parent}
del parent, _smallest

if person_of:
    person_map = pl.DataFrame(
        {"user_id": list(person_of.keys()), "person": list(person_of.values())},
        schema={"user_id": pl.String, "person": pl.String},
    )
else:
    person_map = pl.DataFrame(schema={"user_id": pl.String, "person": pl.String})
del person_of

# --------------------------------------------------------------------------
# Timestamps as Polars expressions. F1/F2/F3 are converted arithmetically
# on string slices; a full-shape ASCII regex plus calendar/range checks
# decide validity, mirroring the strict parser in pipeline.py. Casts are
# never trusted alone (Polars would accept "+5"); every numeric field is
# only used when the shape regex matched, which pins ASCII digits.
# --------------------------------------------------------------------------

F1_RE = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
F2_RE = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}[+-][0-9]{2}:[0-9]{2}$"
F3_RE = r"^[0-9]+$"
NULL_I64 = pl.lit(None, dtype=pl.Int64)


def _int_slice(s, offset, length):
    return s.str.slice(offset, length).cast(pl.Int64, strict=False)


def _civil_ms_expr(y, mo, d, hh, mi, ss):
    """Days-from-civil (Howard Hinnant) + time of day, as expressions."""
    yy = y - (mo <= 2).cast(pl.Int64)
    era = yy // 400
    yoe = yy - era * 400
    mp = (mo + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    days = era * 146097 + doe - 719468
    return days * DAY_MS + ((hh * 60 + mi) * 60 + ss) * 1000


def _calendar_valid(y, mo, d, hh, mi, ss):
    leap = ((y % 4 == 0) & ((y % 100 != 0) | (y % 400 == 0))).fill_null(False)
    dim = (
        pl.when(mo == 2)
        .then(28 + leap.cast(pl.Int64))
        .when((mo == 4) | (mo == 6) | (mo == 9) | (mo == 11))
        .then(30)
        .otherwise(31)
    )
    valid = (
        (y >= 1) & (mo >= 1) & (mo <= 12) & (d >= 1) & (d <= dim)
        & (hh >= 0) & (hh <= 23) & (mi >= 0) & (mi <= 59)
        & (ss >= 0) & (ss <= 59)
    )
    return valid.fill_null(False)


def _ts_parts(s):
    return (
        _int_slice(s, 0, 4), _int_slice(s, 5, 2), _int_slice(s, 8, 2),
        _int_slice(s, 11, 2), _int_slice(s, 14, 2), _int_slice(s, 17, 2),
    )


def _f1_expr(col):
    s = pl.col(col)
    y, mo, d, hh, mi, ss = _ts_parts(s)
    valid = _calendar_valid(y, mo, d, hh, mi, ss) & s.str.contains(F1_RE)
    return pl.when(valid).then(_civil_ms_expr(y, mo, d, hh, mi, ss)).otherwise(NULL_I64)


def _f2_expr(col):
    s = pl.col(col)
    y, mo, d, hh, mi, ss = _ts_parts(s)
    oh, om = _int_slice(s, 20, 2), _int_slice(s, 23, 2)
    off_valid = ((oh >= 0) & (oh <= 23) & (om >= 0) & (om <= 59)).fill_null(False)
    off_ms = (oh * 3600 + om * 60) * 1000
    signed = pl.when(s.str.slice(19, 1) == "+").then(off_ms).otherwise(-off_ms)
    valid = (
        _calendar_valid(y, mo, d, hh, mi, ss)
        & off_valid
        & s.str.contains(F2_RE)
    )
    return (
        pl.when(valid)
        .then(_civil_ms_expr(y, mo, d, hh, mi, ss) - signed)
        .otherwise(NULL_I64)
    )


def _stripped_value(s, max_digits):
    """Integer value of an ASCII digit string, ignoring leading zeros;
    null when the significant part has more than max_digits digits (the
    value is then far outside every range the rules accept)."""
    stripped = s.str.replace(r"^0+", "")
    return (
        pl.when(stripped == "")
        .then(pl.lit(0, dtype=pl.Int64))
        .when(stripped.str.len_chars() <= max_digits)
        .then(stripped.cast(pl.Int64, strict=False))
        .otherwise(NULL_I64)
    )


def _f3_expr(col):
    s = pl.col(col)
    # Any value with 17+ significant digits exceeds every window the
    # generator writes (and 2^63); the Python parser would still parse it
    # and then drop the row as outside the window, so null is equivalent.
    return pl.when(s.str.contains(F3_RE)).then(_stripped_value(s, 16)).otherwise(NULL_I64)


def _ts_ms_expr(col):
    """Parse a ts column into epoch ms (null when bad). The dispatch is
    exactly pipeline.py's parser: F1 = 20 chars ending in Z, F2 = 25 chars
    with a colon at 22 and a sign at 19, anything else can only be F3."""
    s = pl.col(col)
    length = s.str.len_chars()
    is_f1 = (length == 20) & (s.str.slice(19, 1) == "Z")
    is_f2 = (
        (length == 25)
        & (s.str.slice(22, 1) == ":")
        & s.str.slice(19, 1).is_in(["+", "-"])
    )
    return (
        pl.when(is_f1)
        .then(_f1_expr(col))
        .when(is_f2)
        .then(_f2_expr(col))
        .otherwise(_f3_expr(col))
    )


def _ingest_ms_expr(col):
    """ingest_ts is always valid F1 per the rules, so no validity mask."""
    s = pl.col(col)
    return _civil_ms_expr(*_ts_parts(s))


# --------------------------------------------------------------------------
# Events: parse -> drop empty event_id -> dedup on event_id (smallest
# ingest_ts, before validation) -> validate -> drop late (> 24 h).
# --------------------------------------------------------------------------

event_files = sorted(glob.glob(os.path.join(DATA, "events", "*.csv")))
print(f"{len(event_files)} event files")
event_parts = _parse_to_parts(event_files, 7, "events")

# Dedup on a narrow keys frame: (event_id, ingest_ms, _ord) has a total
# order, so after sorting, keep-first is the copy with the smallest
# ingest_ts, ties resolved to the earliest-read copy, exactly like the
# dict merge in pipeline.py.
keys = (
    _scan_parts(event_parts, 7)
    .filter(pl.col("c0") != "")
    .select("c0", "c3", "_ord")
    .with_columns(_ingest_ms_expr("c3").alias("ingest_ms"))
    .select("c0", "ingest_ms", "_ord")
    .sort(["c0", "ingest_ms", "_ord"])
    .collect(engine="streaming")
)
winners = keys.unique(subset=["c0"], keep="first", maintain_order=True).select("_ord")
del keys

_price_val = _stripped_value(pl.col("c6"), 8)
_price_ok = (
    pl.col("c6").str.contains(F3_RE)
    & _price_val.is_not_null()
    & (_price_val >= 1)
    & (_price_val <= 10_000_000)
)
events = (
    _scan_parts(event_parts, 7)
    .join(winners.lazy(), on="_ord", how="semi")
    .with_columns(
        _ts_ms_expr("c2").alias("ts_ms"),
        _ingest_ms_expr("c3").alias("ingest_ms"),
        _price_val.alias("_price_val"),
        _price_ok.alias("_price_ok"),
    )
    .filter(
        (pl.col("c1") != "")
        & (pl.col("c5") != "")
        & pl.col("c4").is_in(["view", "cart", "purchase"])
        & pl.col("ts_ms").is_not_null()
        & (pl.col("ts_ms") >= window_start)
        & (pl.col("ts_ms") < window_end)
        & ((pl.col("c4") != "purchase") | pl.col("_price_ok"))
        & ((pl.col("ingest_ms") - pl.col("ts_ms")) <= DAY_MS)
    )
    .join(person_map.lazy(), left_on="c1", right_on="user_id", how="left")
    .join(products_frame.lazy(), left_on="c5", right_on="product_id", how="left")
    .select(
        person=pl.coalesce([pl.col("person"), pl.col("c1")]),
        ts=pl.col("ts_ms"),
        event_type=pl.col("c4"),
        category=pl.col("category").fill_null("unknown"),
        price=pl.when(pl.col("c4") == "purchase")
        .then(pl.col("_price_val"))
        .otherwise(0),
    )
    .collect(engine="streaming")
)
del winners

# --------------------------------------------------------------------------
# Sessionize per person: sorted by ts, a new session starts when the gap
# from the previous event is over 30 minutes. (Same code as pipeline.py,
# with person/event_type held as categoricals to narrow the sort.)
# --------------------------------------------------------------------------

if events.height:
    events = events.with_columns(
        pl.col("person").cast(pl.Categorical),
        pl.col("event_type").cast(pl.Categorical),
    )
    events = events.sort(["person", "ts"]).with_columns(
        _new_session=(
            pl.col("ts").shift(1).over("person").is_null()
            | ((pl.col("ts") - pl.col("ts").shift(1).over("person")) > SESSION_GAP_MS)
        )
    )
    events = events.with_columns(
        _session_seq=pl.col("_new_session").cast(pl.Int64).cum_sum().over("person")
    )
    sessions = (
        events.group_by(["person", "_session_seq"], maintain_order=True)
        .agg(
            start=pl.col("ts").min(),
            n_events=pl.len(),
            revenue=pl.col("price").sum(),
            has_purchase=(pl.col("event_type") == "purchase").any(),
        )
        .with_columns(day=pl.col("start").cast(pl.Datetime("ms")).dt.date())
        .with_row_index("sid")
    )
    # Per person/day/category purchase totals, for top_category.
    purchases = events.filter(pl.col("event_type") == "purchase").join(
        sessions.select(["person", "_session_seq", "day"]),
        on=["person", "_session_seq"],
        how="left",
    ).with_columns(pl.col("person").cast(pl.String))
    sessions = sessions.with_columns(pl.col("person").cast(pl.String))
else:
    sessions = pl.DataFrame(
        schema={
            "sid": pl.UInt32,
            "person": pl.String,
            "_session_seq": pl.Int64,
            "start": pl.Int64,
            "n_events": pl.UInt32,
            "revenue": pl.Int64,
            "has_purchase": pl.Boolean,
            "day": pl.Date,
        }
    )
    purchases = pl.DataFrame(
        schema={
            "person": pl.String,
            "ts": pl.Int64,
            "event_type": pl.String,
            "category": pl.String,
            "price": pl.Int64,
            "day": pl.Date,
        }
    )

# --------------------------------------------------------------------------
# user_daily: one row per person and day with at least one session.
# --------------------------------------------------------------------------

user_daily = sessions.group_by(["person", "day"]).agg(
    sessions=pl.len(),
    events=pl.col("n_events").sum(),
    revenue_cents=pl.col("revenue").sum(),
)
if purchases.height:
    top = (
        purchases.group_by(["person", "day", "category"])
        .agg(_total=pl.col("price").sum())
        .sort(
            ["person", "day", "_total", "category"],
            descending=[False, False, True, False],
        )
        .group_by(["person", "day"], maintain_order=True)
        .first()
        .select(["person", "day", pl.col("category").alias("top_category")])
    )
    user_daily = user_daily.join(top, on=["person", "day"], how="left")
else:
    user_daily = user_daily.with_columns(top_category=pl.lit(None, dtype=pl.String))
user_daily = user_daily.rename({"person": "user_id"})

# --------------------------------------------------------------------------
# Clicks: parse -> drop empty click_id -> dedup -> validate. Copies of a
# click are identical, so the earliest-read copy per click_id (via the
# same narrow-keys pattern as events) is the "keep any one" of the rules.
# --------------------------------------------------------------------------

click_files = sorted(glob.glob(os.path.join(DATA, "clicks", "*.csv")))
print(f"{len(click_files)} click files")
click_parts = _parse_to_parts(click_files, 4, "clicks")

ckeys = (
    _scan_parts(click_parts, 4)
    .filter(pl.col("c0") != "")
    .select("c0", "_ord")
    .sort(["c0", "_ord"])
    .collect(engine="streaming")
)
cwinners = ckeys.unique(subset=["c0"], keep="first", maintain_order=True).select("_ord")
del ckeys

clicks = (
    _scan_parts(click_parts, 4)
    .join(cwinners.lazy(), on="_ord", how="semi")
    .with_columns(_ts_ms_expr("c3").alias("ts_ms"))
    .filter(
        (pl.col("c1") != "")
        & pl.col("ts_ms").is_not_null()
        & (pl.col("ts_ms") >= window_start - WEEK_MS)
        & (pl.col("ts_ms") < window_end)
    )
    .join(
        campaigns_frame.select("campaign_id").lazy(),
        left_on="c2",
        right_on="campaign_id",
        how="semi",
    )
    .join(person_map.lazy(), left_on="c1", right_on="user_id", how="left")
    .select(
        person=pl.coalesce([pl.col("person"), pl.col("c1")]),
        click_id=pl.col("c0"),
        campaign_id=pl.col("c2"),
        ts=pl.col("ts_ms"),
    )
    .collect(engine="streaming")
)
del cwinners

# --------------------------------------------------------------------------
# Credit and assists: a session's qualifying clicks are the person's clicks
# with start - 7 days <= click.ts <= start. Credit goes to the latest such
# click (tie: smallest click_id); every distinct campaign among them assists.
# (Same code as pipeline.py.)
# --------------------------------------------------------------------------

sessions = sessions.with_columns(credited=pl.lit("organic", dtype=pl.String))
assist_pairs = pl.DataFrame(
    schema={"campaign_id": pl.String, "day": pl.Date, "assisted_sessions": pl.Int64}
)
if sessions.height and clicks.height:
    session_keys = sessions.select(["sid", "person", "start", "day"])
    # Credit: collapse each person's clicks to one row per timestamp (the
    # smallest click_id, with its campaign), then an asof join finds the
    # latest click at or before each session start; the 7-day filter below
    # is still applied, so this picks exactly the click the rules name.
    collapsed = (
        clicks.sort(["person", "ts", "click_id"])
        .group_by(["person", "ts"], maintain_order=True)
        .first()
        .select(["person", "ts", "click_id", "campaign_id"])
        .sort(["person", "ts"])
    )
    best = (
        session_keys.sort(["person", "start"])
        .join_asof(
            collapsed, left_on="start", right_on="ts", by="person",
            strategy="backward",
        )
        .filter((pl.col("start") - pl.col("ts")) <= WEEK_MS)
        .select(["sid", pl.col("campaign_id").alias("credited")])
    )
    if best.height:
        sessions = (
            sessions.drop("credited")
            .join(best, on="sid", how="left")
            .with_columns(pl.col("credited").fill_null("organic"))
        )
    # Assists: a campaign assists a session when one of the person's clicks
    # from it lands in [start - 7 days, start], i.e. when start falls in
    # [click.ts, click.ts + 7 days]. Merging each (person, campaign)'s
    # click windows into intervals keeps the union identical while making
    # the range join far smaller than the raw session x click pairs (the
    # same construction as pipeline.py).
    by_campaign = clicks.sort(["person", "campaign_id", "ts"]).with_columns(
        _new=(
            pl.col("ts").shift(1).over(["person", "campaign_id"]).is_null()
            | ((pl.col("ts") - pl.col("ts").shift(1).over(["person", "campaign_id"]))
               > WEEK_MS)
        )
    )
    intervals = (
        by_campaign.with_columns(
            _iv=pl.col("_new").cast(pl.Int64).cum_sum().over(["person", "campaign_id"])
        )
        .group_by(["person", "campaign_id", "_iv"])
        .agg(lo=pl.col("ts").min(), hi=pl.col("ts").max() + WEEK_MS)
        .select(["person", "campaign_id", "lo", "hi"])
    )
    # The range join's transient is several times its (already large)
    # output, so it runs in person-hash chunks of ~400k sessions and each
    # chunk is aggregated to (campaign, day) counts before the next one
    # starts. Chunking by person keeps every chunk's pairs complete: a
    # session's assists depend only on its own person's clicks. A single
    # person's pair set is the floor, and even the shared-device mega
    # person stays two orders of magnitude below the cap.
    n_chunks = max(1, min(32, -(-session_keys.height // 400_000)))
    assist_counts = []
    for chunk in range(n_chunks):
        sess_k = session_keys.filter(
            (pl.col("person").hash() % n_chunks) == chunk
        )
        if not sess_k.height:
            continue
        intervals_k = intervals.join(
            sess_k.select("person").unique(), on="person", how="semi"
        )
        assisted = sess_k.join_where(
            intervals_k,
            (pl.col("person") == pl.col("person_right"))
            & (pl.col("start") >= pl.col("lo"))
            & (pl.col("start") <= pl.col("hi")),
        )
        if assisted.height:
            assist_counts.append(
                assisted.select(["sid", "campaign_id", "day"])
                .unique()
                .group_by(["campaign_id", "day"])
                .agg(assisted_sessions=pl.len())
            )
    if assist_counts:
        assist_pairs = (
            pl.concat(assist_counts, how="vertical")
            .group_by(["campaign_id", "day"])
            .agg(assisted_sessions=pl.col("assisted_sessions").sum())
        )

# --------------------------------------------------------------------------
# campaign_daily: one row per (campaign, day) with a credited or assisted
# session; pure-assist rows have 0 in the credited columns.
# --------------------------------------------------------------------------

credited_daily = sessions.group_by(["credited", "day"]).agg(
    sessions=pl.len(),
    purchasing_sessions=pl.col("has_purchase").cast(pl.Int64).sum(),
    revenue_cents=pl.col("revenue").sum(),
)
campaign_daily = credited_daily.join(
    assist_pairs, left_on=["credited", "day"], right_on=["campaign_id", "day"],
    how="full", coalesce=True,
).rename({"credited": "campaign_id"})
campaign_daily = campaign_daily.with_columns(
    pl.col("sessions").fill_null(0),
    pl.col("purchasing_sessions").fill_null(0),
    pl.col("revenue_cents").fill_null(0),
    pl.col("assisted_sessions").fill_null(0),
)
channel_frame = campaigns_frame.select(["campaign_id", "channel"]).vstack(
    pl.DataFrame(
        {"campaign_id": ["organic"], "channel": ["organic"]},
        schema={"campaign_id": pl.String, "channel": pl.String},
    )
)
campaign_daily = campaign_daily.join(channel_frame, on="campaign_id", how="left")

# --------------------------------------------------------------------------
# Write both tables with the exact schemas.
# --------------------------------------------------------------------------

USER_DAILY = {
    "user_id": pl.String, "day": pl.Date, "sessions": pl.Int64,
    "events": pl.Int64, "revenue_cents": pl.Int64, "top_category": pl.String,
}
CAMPAIGN_DAILY = {
    "campaign_id": pl.String, "channel": pl.String, "day": pl.Date,
    "sessions": pl.Int64, "purchasing_sessions": pl.Int64, "revenue_cents": pl.Int64,
    "assisted_sessions": pl.Int64,
}

outputs = {
    "user_daily": user_daily.select(list(USER_DAILY)).cast(USER_DAILY),
    "campaign_daily": campaign_daily.select(list(CAMPAIGN_DAILY)).cast(CAMPAIGN_DAILY),
}
for name, frame in outputs.items():
    folder = os.path.join(OUT, name)
    os.makedirs(folder, exist_ok=True)
    frame.write_parquet(os.path.join(folder, "part-0.parquet"))
print(f"wrote user_daily ({outputs['user_daily'].height} rows) and "
      f"campaign_daily ({outputs['campaign_daily'].height} rows)")
