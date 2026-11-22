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
  with vectorized string expressions: comma count for the field count, one
  structured line regex for quote legality (a quote is legal only as one
  pair wrapping a whole field), and a single replace_all to unwrap the
  wrapped fields on lines that passed.
  Valid rows land in Parquet parts under SCRATCH with a global row number
  (_ord, in read order), so parse memory is bounded by the block size and
  the largest single block frame, never by the dataset or file size.
- Dedup: a narrow keys frame (event_id, ingest_ms, _ord) is sorted with
  the streaming engine and keep-first picks the winning copy; the full
  rows are then fetched with a semi join inside the same streaming plan
  that validates them, so the wide raw frame is never materialized.
- Timestamps: F1/F2 conversion uses Polars' native datetime parser,
  fenced by a full-shape ASCII regex and slice range gates (the parser
  alone would accept a leap-second :60 and year 0000, which the rules
  reject); F3 is a gated integer cast. Differentially verified against
  the strict parser's semantics on every ts/ingest string of the
  dev/3M/6M sets plus synthetic edge cases: zero differences.
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
import io
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
    """Yield (lines, has_special) pairs: lists of decoded lines (split on
    b"\\n" only, so a \\r stays an ordinary trailing character of the last
    field), plus whether the block's raw bytes contain a \\r or a BOM.
    The byte test is exact for valid UTF-8: 0x0D is never part of a
    multi-byte sequence, and the byte triple EF BB BF is exactly U+FEFF.
    A line's bytes lie entirely in the buffer it is split from (the tail
    carried over is rechecked with the next buffer), so has_special is
    False only when no yielded line contains either character; it can be
    True spuriously when only the carried tail has one, which merely
    costs the slower path. The final yield is the tail after the last
    newline, or [""] when the file ends with one; an empty line is
    malformed and dropped downstream either way."""
    with open(path, "rb") as handle:
        leftover = b""
        while True:
            chunk = handle.read(_BLOCK_BYTES)
            if not chunk:
                break
            buf = leftover + chunk
            special = b"\r" in buf or b"\xef\xbb\xbf" in buf
            parts = buf.split(b"\n")
            leftover = parts.pop()
            if parts:
                yield [part.decode("utf-8") for part in parts], special
        yield [leftover.decode("utf-8")], (
            b"\r" in leftover or b"\xef\xbb\xbf" in leftover
        )


# A line is dialect-valid when every comma-separated field is either
# quote-free or one cleanly wrapped pair; since values never hold commas,
# wrapped content excludes commas too, so the regex decomposition of the
# line is exactly its field decomposition. On a valid line the quote
# pairs are precisely the wrapped fields, so one replace_all unwraps
# them all. (Verified byte-identical to the per-field checks this
# replaced, on the fixtures, the dev set, and adversarial fuzz lines.)
_LINE_VALID_RE = r'^([^",]*|"[^",]*")(,([^",]*|"[^",]*"))*$'
_UNWRAP_RE = r'"([^"]*)"'


def _lines_frame_expr(lines, schema):
    """Field decomposition of dialect-valid lines via string expressions
    (the original path; kept for the lines the CSV fast path must not
    touch)."""
    df = pl.DataFrame({"line": pl.Series("line", lines, dtype=pl.String)})
    df = df.with_columns(
        _fields=pl.col("line").str.replace_all(_UNWRAP_RE, "$1").str.split(",")
    )
    # The comma-count filter guarantees exactly len(schema) parts per line.
    return df.select(
        pl.col("_fields").list.to_struct(fields=list(schema))
    ).unnest("_fields")


def _lines_frame(lines, nfields, has_special=False):
    """Structurally valid rows of one block, as string columns c0..c{n-1}.
    has_special is the block-level byte check from _line_blocks: when it
    is False no line needs the expression path (see below)."""
    schema = {f"c{i}": pl.String for i in range(nfields)}
    if not lines:
        return pl.DataFrame(schema=schema)
    df = pl.DataFrame({"line": pl.Series("line", lines, dtype=pl.String)})
    # Wrong field count (an empty line included) is malformed; the line
    # regex then checks quote legality. One filter call, so the block is
    # materialized once.
    df = df.filter(
        (pl.col("line").str.count_matches(",") == nfields - 1)
        & pl.col("line").str.contains(_LINE_VALID_RE)
    )
    if df.height == 0:
        return pl.DataFrame(schema=schema)
    # Fast path: the native CSV parser decomposes a dialect-valid line
    # exactly like the expression path (verified byte-identical on the
    # fixtures, the dev set, and ~96k adversarial fuzz lines), with two
    # exceptions, both routed to the expression path line by line: the
    # CSV reader strips a trailing \r that the spec counts as part of
    # the last field, and it strips a leading \ufeff. Its one other
    # difference, null for an unquoted empty field, is normalized back
    # to "" by fill_null. Runs of fast-path lines keep the block's
    # original row order. Most blocks hold no special line at all (the
    # byte-level has_special from _line_blocks), and those skip the
    # per-line Python dispatch entirely: one read_csv over the block.
    if not has_special:
        blob = ("\n".join(df["line"].to_list()) + "\n").encode("utf-8")
        return pl.read_csv(
            io.BytesIO(blob), has_header=False, schema=schema,
            quote_char='"',
        ).fill_null("")
    valid = df["line"].to_list()
    frames = []
    run = []

    def flush_run():
        if run:
            blob = ("\n".join(run) + "\n").encode("utf-8")
            frames.append(
                pl.read_csv(
                    io.BytesIO(blob), has_header=False, schema=schema,
                    quote_char='"',
                ).fill_null("")
            )
            run.clear()

    for line in valid:
        if "\r" in line or "\ufeff" in line:
            flush_run()
            frames.append(_lines_frame_expr([line], schema))
        else:
            run.append(line)
    flush_run()
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="vertical")


def _iter_file_frames(path, nfields):
    """Yield the structurally valid rows of one file, block by block
    (first line is the header). A generator, so callers never hold more
    than one block frame per file."""
    header_pending = True
    for lines, special in _line_blocks(path):
        if header_pending:
            lines = lines[1:]
            header_pending = False
        if lines:
            yield _lines_frame(lines, nfields, special)


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
                # Scratch parts are written and re-read twice within the
                # run; skipping compression and statistics trades a few
                # hundred MB of scratch disk for noticeably less CPU.
                frame.write_parquet(
                    part, compression="uncompressed", statistics=False
                )
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
for a, b in zip(links["c0"].to_list(), links["c1"].to_list()):
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
# Timestamps as Polars expressions. A full-shape ASCII regex plus
# slice range checks decide validity, mirroring the strict parser in
# pipeline.py; the native datetime parser does the F1/F2 conversion
# inside those fences, and F3 is a gated cast. Casts are never trusted
# alone (Polars would accept "+5"); every numeric field is only used
# when the shape regex matched, which pins ASCII digits.
# --------------------------------------------------------------------------

ISO_RE = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(Z|[+-][0-9]{2}:[0-9]{2})$"
NULL_I64 = pl.lit(None, dtype=pl.Int64)


def _int_slice(s, offset, length):
    return s.str.slice(offset, length).cast(pl.Int64, strict=False)


def _ts_parts(s):
    return (
        _int_slice(s, 0, 4), _int_slice(s, 5, 2), _int_slice(s, 8, 2),
        _int_slice(s, 11, 2), _int_slice(s, 14, 2), _int_slice(s, 17, 2),
    )


def _digits_value(s):
    """Integer value of an ASCII digit string (leading zeros allowed), or
    null. The cast alone would accept a leading '+'/'-' or surrounding
    spaces, so the first and last characters must be digits too. A digit
    string too long for an Int64 casts to null, and one with 17-19
    significant digits yields a huge value; every caller range-checks
    (timestamps against the window, prices against 1..10,000,000), and
    the strict parser's value for such strings is likewise far outside
    every accepted range, so the rows are dropped identically either way.
    Fuzz-verified against pipeline.py's parse_ts on ~200k strings."""
    v = s.cast(pl.Int64, strict=False)
    first = s.str.slice(0, 1)
    last = s.str.slice(s.str.len_chars() - 1, 1)
    ok = (
        v.is_not_null()
        & (first >= "0") & (first <= "9")
        & (last >= "0") & (last <= "9")
    )
    return pl.when(ok).then(v).otherwise(NULL_I64)


def _ts_ms_expr(col):
    """Parse a ts column into epoch ms (null when bad): one ISO regex
    dispatches between the F1/F2 conversion and F3 epoch digits,
    mirroring pipeline.py's parser (F1 = Z-terminated, F2 = signed
    offset, anything else can only be F3). The ISO branch uses the
    native datetime parser, fenced by the shape regex plus slice range
    gates: the native parser on its own would accept a leap-second :60
    and year 0000, which the rules reject, so the gates pin y >= 1 and
    the month/day/hour/minute/second ranges and the offset range
    (hour <= 23, minute <= 59); impossible calendar dates (Feb 30,
    Feb 29 in a non-leap year) fail the parse itself. Differentially
    verified against the previous arithmetic conversion on all 13.1M
    ts strings of the dev/3M/6M sets plus synthetic edge cases (leap
    seconds, offset bounds, year 0000/0001/9999, F3 overflow): zero
    differences."""
    s = pl.col(col)
    is_f2 = s.str.len_chars() == 25
    y, mo, d, hh, mi, ss = _ts_parts(s)
    oh, om = _int_slice(s, 20, 2), _int_slice(s, 23, 2)
    in_range = (
        (y >= 1) & (mo >= 1) & (mo <= 12) & (d >= 1) & (d <= 31)
        & (hh >= 0) & (hh <= 23) & (mi >= 0) & (mi <= 59)
        & (ss >= 0) & (ss <= 59)
    ).fill_null(False)
    off_in_range = (~is_f2) | (
        ((oh >= 0) & (oh <= 23) & (om >= 0) & (om <= 59)).fill_null(False)
    )
    ms = pl.when(is_f2).then(
        s.str.to_datetime("%Y-%m-%dT%H:%M:%S%:z", strict=False).cast(pl.Int64)
        // 1000
    ).otherwise(
        s.str.to_datetime("%Y-%m-%dT%H:%M:%SZ", strict=False).cast(pl.Int64)
        // 1000
    )
    iso_ms = (
        pl.when(in_range & off_in_range & ms.is_not_null())
        .then(ms)
        .otherwise(NULL_I64)
    )
    return (
        pl.when(s.str.contains(ISO_RE))
        .then(iso_ms)
        .otherwise(_digits_value(s))
    )


def _ingest_ms_expr(col):
    """ingest_ts is always valid F1 per the rules, so the native parser
    needs no validity mask. Differentially verified against the
    previous arithmetic conversion on all 10.6M ingest strings of the
    dev/3M/6M sets: zero differences."""
    return (
        pl.col(col)
        .str.to_datetime("%Y-%m-%dT%H:%M:%SZ", strict=False)
        .cast(pl.Int64)
        // 1000
    )


# --------------------------------------------------------------------------
# Events: parse -> drop empty event_id -> dedup on event_id (smallest
# ingest_ts, before validation) -> validate -> drop late (> 24 h).
# --------------------------------------------------------------------------

event_files = sorted(glob.glob(os.path.join(DATA, "events", "*.csv")))
print(f"{len(event_files)} event files")
event_parts = _parse_to_parts(event_files, 7, "events")

# Dedup on a narrow keys frame: per event_id, the winner is the copy
# with the smallest (ingest_ts, _ord) -- smallest ingest_ts, ties
# resolved to the earliest-read copy, exactly like the dict merge in
# pipeline.py. The comparison keys on the raw ingest string: ingest_ts
# is always valid F1 (fixed width, zero padded), so string order is
# chronological order, and equal strings are equal instants. A hash
# group_by picks the same winner as sort + keep-first (verified on the
# full dev keys) at a fraction of the cost.
keys = (
    _scan_parts(event_parts, 7)
    .filter(pl.col("c0") != "")
    .select("c0", "c3", "_ord")
    .collect(engine="streaming")
)
winners = keys.group_by("c0").agg(
    pl.col("_ord").sort_by(["c3", "_ord"]).first()
).select("_ord")
del keys

events = (
    _scan_parts(event_parts, 7)
    .join(winners.lazy(), on="_ord", how="semi")
    .with_columns(
        _ts_ms_expr("c2").alias("ts_ms"),
        _ingest_ms_expr("c3").alias("ingest_ms"),
    )
    .filter(
        (pl.col("c1") != "")
        & (pl.col("c5") != "")
        & pl.col("c4").is_in(["view", "cart", "purchase"])
        & pl.col("ts_ms").is_not_null()
        & (pl.col("ts_ms") >= window_start)
        & (pl.col("ts_ms") < window_end)
        & ((pl.col("ingest_ms") - pl.col("ts_ms")) <= DAY_MS)
    )
    .join(person_map.lazy(), left_on="c1", right_on="user_id", how="left")
    .join(products_frame.lazy(), left_on="c5", right_on="product_id", how="left")
    .select(
        person=pl.coalesce([pl.col("person"), pl.col("c1")]),
        ts=pl.col("ts_ms"),
        event_type=pl.col("c4"),
        category=pl.col("category").fill_null("unknown"),
        price_str=pl.col("c6"),
    )
    .collect(engine="streaming")
)
del winners

# Purchase prices are validated only here, on the purchase subset (a few
# percent of events): the rules ignore price_cents for view/cart, and a
# purchase whose price is not ASCII digits with value 1..10,000,000 is
# dropped. Doing it after the collect keeps the price conversion out of
# the full-width streaming plan above.
if events.height:
    _purchases = (
        events.filter(pl.col("event_type") == "purchase")
        .with_columns(_digits_value(pl.col("price_str")).alias("price"))
        .filter(
            pl.col("price").is_not_null()
            & (pl.col("price") >= 1)
            & (pl.col("price") <= 10_000_000)
        )
        .drop("price_str")
    )
    events = pl.concat(
        [
            events.filter(pl.col("event_type") != "purchase")
            .with_columns(price=pl.lit(0, dtype=pl.Int64))
            .drop("price_str"),
            _purchases,
        ],
        how="vertical",
    )
    del _purchases
else:
    events = events.with_columns(price=pl.lit(0, dtype=pl.Int64)).drop("price_str")

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
    events = events.sort(["person", "ts"])
    # Session boundaries without window functions: the frame is sorted
    # by (person, ts), so a new session starts exactly where the person
    # changes (compared by categorical code, which is one-to-one with
    # the string) or the gap from the previous row exceeds 30 minutes.
    # A running total over the whole frame then numbers the sessions
    # 0..N-1 in row order -- the same order the old group_by +
    # with_row_index produced, so the repeat_by day alignment below
    # still lines up.
    _code = pl.col("person").to_physical()
    events = events.with_columns(
        _sid=(
            _code.shift(1).is_null()
            | (_code != _code.shift(1))
            | ((pl.col("ts") - pl.col("ts").shift(1)) > SESSION_GAP_MS)
        ).cast(pl.Int64).cum_sum()
        - 1
    )
    sessions = (
        events.group_by("_sid", maintain_order=True)
        .agg(
            person=pl.col("person").first(),
            start=pl.col("ts").min(),
            n_events=pl.len(),
            revenue=pl.col("price").sum(),
            has_purchase=(pl.col("event_type") == "purchase").any(),
        )
        .rename({"_sid": "sid"})
        .with_columns(
            pl.col("sid").cast(pl.UInt32),
            day=pl.col("start").cast(pl.Datetime("ms")).dt.date(),
        )
    )
    # Per person/day/category purchase totals, for top_category. Events
    # are sorted by (person, ts) and sessions are ts-contiguous blocks in
    # (person, _session_seq) order (the group_by above keeps first-
    # appearance order), so repeating each session's day by its event
    # count aligns the day onto its events with no join.
    events = events.with_columns(
        _session_day=sessions["day"].repeat_by(sessions["n_events"]).explode()
    )
    purchases = (
        events.filter(pl.col("event_type") == "purchase")
        .select(["person", "ts", "event_type", "category", "price",
                 pl.col("_session_day").alias("day")])
        .with_columns(pl.col("person").cast(pl.String))
    )
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
    .collect(engine="streaming")
)
cwinners = ckeys.group_by("c0").agg(pl.col("_ord").min()).select("_ord")
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
    by_campaign = clicks.sort(["person", "campaign_id", "ts"])
    # Interval boundaries without window functions: sorted by
    # (person, campaign, ts), a new interval starts where the person or
    # the campaign changes or the gap from the previous click exceeds
    # the 7-day window (the same chaining rule as before: windows merge
    # while consecutive clicks are at most a week apart); a frame-wide
    # running total numbers the intervals.
    _p = pl.col("person")
    _c = pl.col("campaign_id")
    by_campaign = by_campaign.with_columns(
        _iv=(
            _p.shift(1).is_null()
            | (_p != _p.shift(1))
            | (_c != _c.shift(1))
            | ((pl.col("ts") - pl.col("ts").shift(1)) > WEEK_MS)
        ).cast(pl.Int64).cum_sum()
    )
    intervals = (
        by_campaign.group_by("_iv")
        .agg(
            person=pl.col("person").first(),
            campaign_id=pl.col("campaign_id").first(),
            lo=pl.col("ts").min(),
            hi=pl.col("ts").max() + WEEK_MS,
        )
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
    # Split sessions and intervals into person-hash chunks once, up
    # front (the old loop re-filtered session_keys and re-semi-joined
    # intervals for every chunk). A person's chunk comes from the same
    # hash expression in both splits, and the intervals split joins on
    # the person string itself, so every chunk still holds complete
    # per-person pairs.
    sess_chunks = session_keys.with_columns(
        _chunk=(pl.col("person").hash() % n_chunks).cast(pl.Int64)
    ).partition_by("_chunk", as_dict=True)
    person_chunk = session_keys.select("person").unique().with_columns(
        _chunk=(pl.col("person").hash() % n_chunks).cast(pl.Int64)
    )
    interval_chunks = {}
    for _key, _frame in intervals.join(
        person_chunk, on="person", how="inner"
    ).partition_by("_chunk", as_dict=True).items():
        interval_chunks[_key[0] if isinstance(_key, tuple) else _key] = _frame
    assist_counts = []
    for _key, sess_k in sess_chunks.items():
        chunk = _key[0] if isinstance(_key, tuple) else _key
        if not sess_k.height:
            continue
        intervals_k = interval_chunks.get(chunk)
        if intervals_k is None or not intervals_k.height:
            continue
        assisted = sess_k.join_where(
            intervals_k,
            (pl.col("person") == pl.col("person_right"))
            & (pl.col("start") >= pl.col("lo"))
            & (pl.col("start") <= pl.col("hi")),
        )
        if assisted.height:
            assist_counts.append(
                assisted.select(["sid", "campaign_id", "day"])
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
