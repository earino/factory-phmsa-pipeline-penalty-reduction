import hashlib
import io
import re
from pathlib import Path

import pandas as pd

from source_io import fetch

RAW_URL = 'https://primis.phmsa.dot.gov/enforcement-documents/PHMSA%20Pipeline%20Enforcement%20Raw%20Data.txt'
RAW_NAME = 'phmsa_pipeline_enforcement_raw.txt'
TRAIN_START = pd.Timestamp('2002-01-01')
TRAIN_END = pd.Timestamp('2017-01-01')
EVAL_END = pd.Timestamp('2020-01-01')
HOLDOUT_END = pd.Timestamp('2023-07-01')
HORIZON_MONTHS = 36
OUT_COLUMNS = ['case_id', 'notice_date', 'proposed_penalty_usd', 'violation_category_count', 'cited_regulation_count', 'integrity_management_flag', 'pipeline_type', 'region', 'discovery_route', 'notice_actions', 'proposed_compliance_order', 'penalty_reduced']
FEATURES = OUT_COLUMNS[2:-1]
MISSING_TOKENS = ('', 'nan', 'none', 'null', 'n/a', 'na', '-', '--', 'nat', 'unknown', 'not available')
DATE_FORMATS = ('%m/%d/%y', '%m/%d/%Y', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%Y/%m/%d')
ALIASES = {
    'wms': ['WMS_Activity_Seq', 'WMS Activity Seq', 'WMS_ACTIVITY_SEQ'],
    'cpf': ['CPF_Number', 'CPF Number', 'CPF', 'Case_Number', 'Case Number'],
    'case_type': ['Case_Type', 'Case Type'],
    'notice': ['Notice_Sent_Date', 'Notice Sent Date'],
    'proposed': ['Proposed_Penalties', 'Proposed Penalties'],
    'assessed': ['Assessed_Penalties', 'Assessed Penalties'],
    'closed': ['Closed_Date', 'Closed Date'],
    'violation_category': ['Violation_Category', 'Violation Category', 'Proposed_Violations', 'Proposed Violations'],
    'cited': ['Cited_Regulations', 'Cited Regulations', 'Proposed_Cited_Regulations', 'Proposed Cited Regulations'],
    'pipeline_type': ['Pipeline_Type', 'Pipeline Type', 'System_Type', 'System Type', 'Segment_Type', 'Segment Type'],
    'region': ['Region', 'PHMSA_Region', 'PHMSA Region', 'Region_Code', 'Region Code'],
    'discovery': ['How_Discovered?', 'How_Discovered', 'How Discovered', 'Discovery_Route', 'Discovery Route'],
    'notice_actions': ['Notice_Actions', 'Notice Actions', 'Notice_Action', 'Notice Action'],
    'compliance_order': ['Proposed_Compliance_Order_Ind', 'Proposed Compliance Order Ind', 'Proposed_Compliance_Order', 'Proposed Compliance Order'],
}


def _norm(value):
    return ''.join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded(message):
    return str(message)[:4000]


def _add_check(checks, name, observed, expected, passed, kind=None):
    item = {'check': str(name), 'observed': str(observed)[:1500], 'expected': str(expected)[:1500], 'passed': bool(passed)}
    if kind is not None:
        item['kind'] = str(kind)
    checks.append(item)
    if not passed:
        raise RuntimeError(_bounded('CHECK FAILED: ' + str(name) + ' | observed: ' + str(observed) + ' | expected: ' + str(expected)))


def _text_series(series):
    return series.fillna('').astype(str).str.strip()


def _missing_mask(text):
    return text.str.lower().isin(list(MISSING_TOKENS))


def _decode_raw(raw_bytes):
    attempts = []
    for encoding in ('utf-8-sig', 'cp1252', 'latin-1'):
        try:
            return raw_bytes.decode(encoding), encoding, attempts
        except UnicodeDecodeError as exc:
            attempts.append(encoding + ' -> ' + str(exc))
    raise RuntimeError(_bounded('Unable to decode publisher text with utf-8-sig, cp1252 or latin-1. Source URL: ' + RAW_URL + '; attempts: ' + ' | '.join(attempts)))


def _resolve_columns(columns, raw_path):
    observed = {}
    for column in columns:
        observed.setdefault(_norm(column), column)
    resolved = {}
    missing = []
    for key, names in ALIASES.items():
        hit = None
        for name in names:
            if _norm(name) in observed:
                hit = observed[_norm(name)]
                break
        if hit is None:
            missing.append(key)
        else:
            resolved[key] = hit
    if missing:
        missing_desc = [key + ' (accepted names: ' + ', '.join(ALIASES[key]) + ')' for key in missing]
        message = 'Required source fields are missing. selected archive member: ' + str(raw_path) + '; source URL: ' + RAW_URL + '; observed headers: ' + ', '.join(str(column) for column in columns) + '; missing names: ' + '; '.join(missing_desc)
        raise RuntimeError(_bounded(message))
    return resolved


def _parse_dates(series):
    text = _text_series(series)
    text = text.mask(_missing_mask(text))
    result = pd.Series(pd.NaT, index=series.index, dtype='datetime64[ns]')
    remaining = text.notna()
    for fmt in DATE_FORMATS:
        if not bool(remaining.any()):
            break
        parsed = pd.to_datetime(text[remaining], format=fmt, errors='coerce')
        good = parsed.notna()
        if bool(good.any()):
            result.loc[parsed.index[good]] = parsed.loc[good]
        remaining = remaining & result.isna()
    if bool(remaining.any()):
        parsed = pd.to_datetime(text[remaining], errors='coerce')
        good = parsed.notna()
        if bool(good.any()):
            result.loc[parsed.index[good]] = parsed.loc[good]
    return result


def _parse_money(series):
    text = _text_series(series)
    negative = text.str.startswith('(') & text.str.endswith(')')
    cleaned = text.str.replace('$', '', regex=False).str.replace(',', '', regex=False)
    cleaned = cleaned.str.replace('(', '', regex=False).str.replace(')', '', regex=False)
    cleaned = cleaned.mask(_missing_mask(text))
    values = pd.to_numeric(cleaned, errors='coerce')
    values = values.where(~negative, -values)
    return values


def _count_tokens(series, distinct=False):
    values = []
    for item in _text_series(series):
        if item.lower() in MISSING_TOKENS:
            values.append(pd.NA)
            continue
        parts = [part.strip() for part in re.split(r'[;,\n]', item) if part.strip()]
        if distinct:
            parts = list(dict.fromkeys(part.lower() for part in parts))
        values.append(len(parts) if parts else pd.NA)
    return pd.Series(values, index=series.index, dtype='Int64')


def _integrity_flag(series):
    values = []
    for item in _text_series(series):
        if item.lower() in MISSING_TOKENS:
            values.append(pd.NA)
        else:
            values.append(1 if 'integrity management' in item.lower() else 0)
    return pd.Series(values, index=series.index, dtype='Int64')


def _indicator(series):
    values = []
    for item in _text_series(series):
        text = item.upper()
        if text in ('Y', 'YES', '1', 'TRUE', 'T'):
            values.append(1)
        elif text in ('N', 'NO', '0', 'FALSE', 'F'):
            values.append(0)
        else:
            values.append(pd.NA)
    return pd.Series(values, index=series.index, dtype='Int64')


def _text_feature(series):
    text = _text_series(series)
    return text.mask(_missing_mask(text))


def _case_type_mask(series):
    text = _text_series(series).str.lower().str.replace(r'[^a-z0-9]+', '', regex=True)
    return text.str.contains('noticeofprobableviolation', regex=False) | text.str.contains('nopv', regex=False)


def build(output: Path, cache: Path):
    output = Path(output)
    cache = Path(cache)
    (output / 'public').mkdir(parents=True, exist_ok=True)
    (output / 'private').mkdir(parents=True, exist_ok=True)
    checks = []
    raw_path = Path(fetch(RAW_URL, cache, RAW_NAME))
    _add_check(checks, 'source file downloaded', raw_path.name + ' with ' + str(raw_path.stat().st_size) + ' bytes', 'nonempty local copy recorded by source_io.fetch together with its source URL and SHA256', raw_path.exists() and raw_path.stat().st_size > 0)
    digest = _sha256(raw_path)
    _add_check(checks, 'source file SHA256 recorded', digest, '64 hexadecimal characters recorded for the selected source member', len(digest) == 64)
    raw_bytes = raw_path.read_bytes()
    non_ascii = int(sum(1 for byte in raw_bytes if byte > 127))
    text, encoding, decode_attempts = _decode_raw(raw_bytes)
    _add_check(checks, 'text decoded without loss', 'encoding ' + encoding + '; non-ASCII bytes ' + str(non_ascii) + '; earlier attempts: ' + (' | '.join(decode_attempts) if decode_attempts else 'none'), 'publisher text decodes losslessly and no byte is replaced with the U+FFFD character', text.count(chr(0xFFFD)) == 0)
    lines = text.splitlines()
    header_line = lines[0] if lines else ''
    _add_check(checks, 'header row is tab delimited', str(header_line.count(chr(9)) + 1) + ' header fields; first 300 characters: ' + header_line[:300], 'tab-separated header row carrying multiple columns', header_line.count(chr(9)) >= 9)
    try:
        raw = pd.read_csv(io.StringIO(text), sep=chr(9), quotechar=chr(34), dtype=str, keep_default_na=False, engine='python', on_bad_lines='error')
    except Exception as exc:
        raise RuntimeError(_bounded('Source member could not be parsed as tab-delimited text. selected archive member: ' + str(raw_path) + '; source URL: ' + RAW_URL + '; decoded with encoding: ' + encoding + '; header line: ' + header_line[:600] + '; parse error: ' + str(exc)))
    columns = list(raw.columns)
    _add_check(checks, 'parsed column count', str(len(columns)) + ' columns; first 200 header characters: ' + header_line[:200], 'at least 10 tab-delimited columns', len(columns) >= 10)
    resolved = _resolve_columns(columns, raw_path)
    case_type_text = _text_series(raw[resolved['case_type']])
    nopv_mask = _case_type_mask(raw[resolved['case_type']])
    type_counts = case_type_text.value_counts(dropna=False).head(20)
    _add_check(checks, 'Case_Type distribution', '; '.join(str(key) + ': ' + str(int(value)) for key, value in type_counts.items()), 'rows whose Case_Type denotes a Notice of Probable Violation are present; every other case type is counted here and excluded rather than treated as a failure', bool(nopv_mask.any()))
    nopv = raw.loc[nopv_mask].copy()
    _add_check(checks, 'NOPV row count', str(len(nopv)) + ' rows; excluded other case types: ' + str(int(len(raw) - len(nopv))), 'at least one row for the frozen population', len(nopv) > 0)
    proposed_all = _parse_money(nopv[resolved['proposed']])
    notice_all = _parse_dates(nopv[resolved['notice']])
    closed_all = _parse_dates(nopv[resolved['closed']])
    assessed_all = _parse_money(nopv[resolved['assessed']])
    proposed_text = _text_series(nopv[resolved['proposed']])
    proposed_missing = _missing_mask(proposed_text)
    proposed_bad = (~proposed_missing) & proposed_all.isna()
    _add_check(checks, 'Proposed_Penalties parse coverage', str(int((~proposed_missing & proposed_all.notna()).sum())) + ' of ' + str(len(nopv)) + ' NOPV rows parse as money; nonblank unparseable rows excluded: ' + str(int(proposed_bad.sum())) + ('; examples: ' + '; '.join(proposed_text[proposed_bad].head(5).tolist()) if bool(proposed_bad.any()) else ''), 'nonblank Proposed_Penalties text parses as a USD amount; rows that cannot be parsed are excluded from the population and counted here', bool((~proposed_missing & proposed_all.notna()).any()))
    notice_missing = int(notice_all.isna().sum())
    _add_check(checks, 'Notice_Sent_Date parse coverage', str(len(notice_all) - notice_missing) + ' of ' + str(len(notice_all)) + ' NOPV rows carry a parsed notice date', 'nonmissing parseable notice dates define the population; rows without one are excluded and counted', (len(notice_all) - notice_missing) > 0)
    positive_mask = proposed_all.notna() & proposed_all.gt(0)
    eligible = nopv.loc[positive_mask & notice_all.notna()].copy()
    _add_check(checks, 'positive proposed penalty population', 'excluded ' + str(int((~positive_mask).sum())) + ' NOPV rows with missing or nonpositive Proposed_Penalties; excluded ' + str(notice_missing) + ' rows with an unparseable notice date; kept ' + str(len(eligible)), 'every kept row has a strictly positive proposed penalty and a parsed notice date, and at least one row is kept', bool(len(eligible) > 0 and eligible.index.isin(nopv.index).all() and eligible[resolved['proposed']].notna().all()))
    eligible['_notice'] = notice_all.loc[eligible.index]
    eligible['_proposed'] = proposed_all.loc[eligible.index]
    eligible['_closed'] = closed_all.loc[eligible.index]
    eligible['_assessed'] = assessed_all.loc[eligible.index]
    _add_check(checks, 'kept proposed penalties strictly positive', str(int(eligible['_proposed'].gt(0).sum())) + ' of ' + str(len(eligible)) + ' kept rows; minimum value ' + str(float(eligible['_proposed'].min())), 'all kept proposed penalty amounts are greater than zero', bool(eligible['_proposed'].gt(0).all()))
    wms_text = _text_series(eligible[resolved['wms']])
    wms_missing = _missing_mask(wms_text)
    _add_check(checks, 'WMS_Activity_Seq usability', 'missing keys among eligible rows: ' + str(int(wms_missing.sum())) + ' of ' + str(len(eligible)) + '; present keys: ' + str(int((~wms_missing).sum())), 'the activity sequence key is present for the deduplicated population; rows lacking it are excluded and counted', bool((~wms_missing).any()))
    eligible = eligible.loc[~wms_missing].copy()
    eligible['_wms'] = _text_series(eligible[resolved['wms']])
    in_window = eligible['_notice'].ge(TRAIN_START) & eligible['_notice'].lt(HOLDOUT_END)
    out_window = int((~in_window).sum())
    eligible = eligible.loc[in_window].copy()
    _add_check(checks, 'notice date population window', 'excluded ' + str(out_window) + ' eligible rows outside 2002-01-01 through 2023-06-30; kept ' + str(len(eligible)) + ' rows from ' + (eligible['_notice'].min().strftime('%Y-%m-%d') if len(eligible) else 'none') + ' to ' + (eligible['_notice'].max().strftime('%Y-%m-%d') if len(eligible) else 'none'), 'every kept notice date falls in the frozen half-open window [2002-01-01, 2023-07-01)', bool(len(eligible) > 0 and eligible['_notice'].ge(TRAIN_START).all() and eligible['_notice'].lt(HOLDOUT_END).all()))
    before_dedup = len(eligible)
    eligible = eligible.sort_values(['_notice', '_wms', resolved['cpf']], kind='mergesort')
    eligible = eligible.drop_duplicates(subset=['_wms'], keep='first').copy()
    _add_check(checks, 'WMS duplicate removal', 'removed ' + str(before_dedup - len(eligible)) + ' repeated WMS_Activity_Seq rows; remaining ' + str(len(eligible)), 'keep the earliest notice per WMS_Activity_Seq so related rows never straddle splits', bool(before_dedup >= len(eligible) and len(eligible) > 0 and not eligible['_wms'].duplicated().any()))
    closed_text = _text_series(eligible[resolved['closed']])
    closed_missing = _missing_mask(closed_text)
    closed_bad = (~closed_missing) & eligible['_closed'].isna()
    _add_check(checks, 'Closed_Date parse', 'unparseable nonblank values among kept rows: ' + str(int(closed_bad.sum())) + ('; examples: ' + '; '.join(closed_text[closed_bad].head(5).tolist()) if bool(closed_bad.any()) else '') + '; blank values: ' + str(int(closed_missing.sum())), 'each nonblank Closed_Date on a kept row parses as a calendar date; blanks are interpreted as not closed', int(closed_bad.sum()) == 0)
    assessed_text = _text_series(eligible[resolved['assessed']])
    assessed_missing = _missing_mask(assessed_text)
    assessed_bad = (~assessed_missing) & eligible['_assessed'].isna()
    _add_check(checks, 'Assessed_Penalties parse', 'unparseable nonblank values among kept rows: ' + str(int(assessed_bad.sum())) + ('; examples: ' + '; '.join(assessed_text[assessed_bad].head(5).tolist()) if bool(assessed_bad.any()) else '') + '; blank values: ' + str(int(assessed_missing.sum())), 'each nonblank Assessed_Penalties on a kept row parses as a USD amount; blanks stay missing and are never turned into a numeric zero', int(assessed_bad.sum()) == 0)
    closed_before = eligible['_closed'].notna() & eligible['_closed'].lt(eligible['_notice'])
    _add_check(checks, 'Closed_Date chronology', 'kept rows with Closed_Date before Notice_Sent_Date: ' + str(int(closed_before.sum())), 'no kept row closes before its notice was sent', int(closed_before.sum()) == 0)
    horizon_end = eligible['_notice'] + pd.DateOffset(months=HORIZON_MONTHS)
    closed_ok = eligible['_closed'].notna() & eligible['_closed'].ge(eligible['_notice']) & eligible['_closed'].le(horizon_end)
    assessed_ok = eligible['_assessed'].notna() & eligible['_assessed'].lt(eligible['_proposed'])
    eligible['penalty_reduced'] = (closed_ok & assessed_ok).astype(int)
    positive_count = int(eligible['penalty_reduced'].sum())
    _add_check(checks, 'label composition', 'positives ' + str(positive_count) + '; negatives ' + str(int(len(eligible) - positive_count)) + '; closed within horizon ' + str(int(closed_ok.sum())) + '; assessed below proposed ' + str(int(assessed_ok.sum())) + '; closed but missing assessment ' + str(int((closed_ok & eligible['_assessed'].isna()).sum())), 'the label is 1 only when a parsed Closed_Date falls inside 36 months of the notice and a parsed Assessed_Penalties is strictly below Proposed_Penalties; censored and unassessed cases stay 0', bool(0 <= positive_count <= len(eligible)) and eligible['penalty_reduced'].isin([0, 1]).all())
    _add_check(checks, 'unit check for the 36 month horizon', 'Closed_Date compared against Notice_Sent_Date advanced by ' + str(HORIZON_MONTHS) + ' calendar months; the semantic and the numeric unit are both months', 'the frozen domain threshold is a month count, not days or years', HORIZON_MONTHS == 36, kind='unit')
    violation_counts = _count_tokens(eligible[resolved['violation_category']], distinct=False)
    cited_counts = _count_tokens(eligible[resolved['cited']], distinct=True)
    integrity = _integrity_flag(eligible[resolved['violation_category']])
    pipeline = _text_feature(eligible[resolved['pipeline_type']])
    region = _text_feature(eligible[resolved['region']])
    discovery = _text_feature(eligible[resolved['discovery']])
    actions = _text_feature(eligible[resolved['notice_actions']])
    compliance = _indicator(eligible[resolved['compliance_order']])
    for label, series in [('violation_category_count', violation_counts), ('cited_regulation_count', cited_counts), ('integrity_management_flag', integrity), ('pipeline_type', pipeline), ('region', region), ('discovery_route', discovery), ('notice_actions', actions), ('proposed_compliance_order', compliance)]:
        observed = int(series.notna().sum())
        _add_check(checks, label + ' coverage', str(observed) + ' of ' + str(len(series)) + ' kept rows nonmissing; distinct values ' + str(int(series.dropna().nunique())), 'the declared feature is parsed from its publisher field for kept rows and is not an empty column; missing values stay missing', observed > 0)
    cpf = _text_series(eligible[resolved['cpf']])
    base_id = cpf.mask(_missing_mask(cpf), eligible['_wms'])
    duplicate_id = base_id.duplicated(keep=False)
    case_id = base_id.where(~duplicate_id, base_id + '-' + eligible['_wms'])
    _add_check(checks, 'case ID completeness', 'blank identifiers: ' + str(int(case_id.eq('').sum())) + ' of ' + str(len(case_id)) + '; identifiers taken from CPF_Number: ' + str(int((~_missing_mask(cpf)).sum())) + '; fallback to WMS_Activity_Seq: ' + str(int(_missing_mask(cpf).sum())), 'every kept row carries a nonblank stable identifier', int(case_id.eq('').sum()) == 0)
    _add_check(checks, 'case ID uniqueness', 'duplicate identifiers after disambiguation: ' + str(int(case_id.duplicated().sum())), 'zero duplicate identifiers so every shipped row ID is unique', int(case_id.duplicated().sum()) == 0)
    out = pd.DataFrame({
        'case_id': case_id,
        'notice_date': eligible['_notice'].dt.strftime('%Y-%m-%d'),
        'proposed_penalty_usd': eligible['_proposed'],
        'violation_category_count': violation_counts,
        'cited_regulation_count': cited_counts,
        'integrity_management_flag': integrity,
        'pipeline_type': pipeline,
        'region': region,
        'discovery_route': discovery,
        'notice_actions': actions,
        'proposed_compliance_order': compliance,
        'penalty_reduced': eligible['penalty_reduced'].astype(int),
    })
    whole_dollars = bool((eligible['_proposed'].dropna() % 1 == 0).all())
    if whole_dollars:
        out['proposed_penalty_usd'] = out['proposed_penalty_usd'].astype('Int64')
    else:
        out['proposed_penalty_usd'] = out['proposed_penalty_usd'].astype('float64')
    stored = pd.to_numeric(out['proposed_penalty_usd'], errors='coerce').astype('float64')
    source_values = eligible['_proposed'].astype('float64')
    _add_check(checks, 'proposed penalty stored without rounding', 'stored dtype ' + str(out['proposed_penalty_usd'].dtype) + '; whole-dollar values present: ' + str(whole_dollars) + '; maximum absolute difference from the parsed source value: ' + str(float((stored - source_values).abs().fillna(0).max())), 'shipped amounts equal the parsed publisher amounts, with decimals kept when the publisher uses them', bool((stored - source_values).abs().fillna(0).le(1e-9).all()))
    split = pd.Series('', index=out.index, dtype='object')
    split[eligible['_notice'].lt(TRAIN_END)] = 'train'
    split[eligible['_notice'].ge(TRAIN_END) & eligible['_notice'].lt(EVAL_END)] = 'eval'
    split[eligible['_notice'].ge(EVAL_END) & eligible['_notice'].lt(HOLDOUT_END)] = 'holdout'
    out['_split'] = split
    _add_check(checks, 'split assignment completeness', '; '.join(name + ': ' + str(int(split.eq(name).sum())) for name in ('train', 'eval', 'holdout')) + '; unassigned: ' + str(int(split.eq('').sum())), 'every kept notice date lands in exactly one frozen half-open window', int(split.eq('').sum()) == 0)
    out = out.sort_values(['notice_date', 'case_id'], kind='mergesort').reset_index(drop=True)
    _add_check(checks, 'output column order', ', '.join(str(column) for column in out.columns), ', '.join(OUT_COLUMNS + ['_split']), list(out.columns) == OUT_COLUMNS + ['_split'])
    _add_check(checks, 'target binary', ', '.join(str(value) for value in sorted(out['penalty_reduced'].unique())), 'only 0 and 1', bool(out['penalty_reduced'].isin([0, 1]).all()))
    _add_check(checks, 'target header name', str(out.columns[OUT_COLUMNS.index('penalty_reduced')]), 'the shipped target column is named exactly penalty_reduced', 'penalty_reduced' in out.columns)
    _add_check(checks, 'no post hoc column shipped', ', '.join(FEATURES), 'Assessed_Penalties and Closed_Date never appear as features', not any(name in out.columns for name in ('Assessed_Penalties', 'Closed_Date', 'Assessed_Penalties_Parsed')))
    for split_name in ('train', 'eval', 'holdout'):
        part = out.loc[out['_split'].eq(split_name)]
        _add_check(checks, split_name + ' split rows', str(len(part)) + ' rows; positives ' + str(int(part['penalty_reduced'].sum())), 'greater than zero rows in the frozen window', len(part) > 0)
        _add_check(checks, split_name + ' split columns', ', '.join(str(column) for column in part.columns), 'identical columns across all splits', list(part.columns) == list(out.columns))
    for column in FEATURES:
        observed = int(out[column].notna().sum())
        _add_check(checks, column + ' final coverage', str(observed) + ' of ' + str(len(out)) + ' rows nonmissing', 'at least one observed value in the shipped column; missing stays missing', observed > 0)
    for split_name, relative in (('train', 'public/train.csv'), ('eval', 'public/eval.csv'), ('holdout', 'private/holdout.csv')):
        part = out.loc[out['_split'].eq(split_name), OUT_COLUMNS].copy()
        path = output / relative
        part.to_csv(path, index=False, lineterminator=chr(10))
        written = pd.read_csv(path, dtype=str, keep_default_na=False)
        _add_check(checks, 'written file ' + relative, str(path.stat().st_size) + ' bytes; ' + str(len(part)) + ' rows; header ' + ', '.join(str(column) for column in part.columns), 'file exists, is nonempty and round-trips its declared header and row count', bool(path.exists() and path.stat().st_size > 0 and list(part.columns) == OUT_COLUMNS and len(part) > 0) and list(written.columns) == OUT_COLUMNS)
    return checks
