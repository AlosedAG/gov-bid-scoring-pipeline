import pandas as pd
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
from rapidfuzz import process, fuzz
from config import *

def fix_date_format(date_str):
    if pd.isna(date_str) or date_str == '':
        return ''
    date_str = str(date_str)
    if 'T' in date_str:
        return date_str.split('T')[0]
    return date_str[:10]

def parse_location_field(loc):
    if not loc or pd.isna(loc):
        return ('', False, '', False)

    s = str(loc).strip().lower()
    
    discard_flag = any(term in s for term in DISCARD_TERMS)
    is_county = 'county' in s

    s = re.sub(r'^(city of|county of)\s+', '', s, flags=re.IGNORECASE).strip()

    if ',' in s:
        parts = s.rsplit(',', 1)
        name = parts[0].strip()
        state = parts[1].strip()
    else:
        parts = s.split()
        if len(parts) > 1:
            name = ' '.join(parts[:-1])
            state = parts[-1].strip()
        else:
            name = s
            state = ''

    name = re.sub(r'\bcounty\b', '', name, flags=re.IGNORECASE).strip()

    if 'city' in name.lower() and not name.lower().endswith('city'):
        name = re.sub(r'\bcity\b', '', name, flags=re.IGNORECASE).strip()
        name = f"{name} City"

    state_up = state.title()
    state_abbr = STATE_ABBR.get(state_up, state_up.upper() if state_up else '')

    return (name.strip().title(), is_county, state_abbr, discard_flag)

def format_location_name(name, is_county, st_abbr):
    if not name:
        return ''
    if is_county:
        return f'{name} County, {st_abbr}' if st_abbr else f'{name} County'
    else:
        return f'{name}, {st_abbr}' if st_abbr else name

def is_aggregator(link):
    if pd.isna(link) or link == '':
        return False
    link_str = str(link).lower()
    for agg_domain in AGG_DOMAINS:
        if agg_domain in link_str:
            return True
    try:
        dom = urlparse(link_str).netloc
        for agg_domain in AGG_DOMAINS:
            if agg_domain in dom:
                return True
    except:
        pass
    return False

def check_date_ok(due_date_col):
    today = datetime.now(timezone.utc)
    parsed = pd.to_datetime(due_date_col, errors='coerce').dt.tz_localize(timezone.utc)
    return parsed >= (today + timedelta(days=7))

def is_gov_related(row):
    desc = str(row.get('Description','')).lower()
    link = str(row.get('Link','')).lower()
    if re.search(r'\b(the city of|city of|city)\b', desc):
        return True
    if re.search(r'\b(the county of|county of|county)\b', desc):
        return True
    try:
        dom = urlparse(link).netloc
        if dom.endswith('.gov'):
            return True
    except:
        pass
    return False

def find_landscape_match(title, landscape_list, threshold=85):
    """
    Uses RapidFuzz token_set_ratio to find matches.
    - token_set_ratio handles shared words well.
    - It penalizes "Test" matching "Testing" (score ~72).
    - It rewards "Pavement" matching "Pavement Data" (score 100).
    """
    if pd.isna(title) or title == '' or not landscape_list:
        return "No match"
    
    # Extract the best matches above the threshold
    # limit=3 allows us to find multiple relevant landscapes
    matches = process.extract(
        title, 
        landscape_list, 
        scorer=fuzz.token_set_ratio, 
        limit=3,
        score_cutoff=threshold
    )
    
    if matches:
        # matches is a list of tuples: (match_string, score, index)
        # We join the matched landscape names
        return ", ".join([m[0] for m in matches])
    else:
        return "No match"