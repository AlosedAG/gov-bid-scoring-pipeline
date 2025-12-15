from config import *
import pandas as pd

def pop_ok(row):
    pop = row.get('Population')
    if pd.isna(pop) or pop == '':
        return False
    try:
        pop = int(pop)
    except:
        return False
    
    if row.get('is_county'):
        return pop >= COUNTY_POP_THRESH
    else:
        return pop >= CITY_POP_THRESH

def assign_rank(row):
    title = str(row.get('Title', '')).lower()
    prob = row.get('software_prob', 0)
    discard_flag = row.get('discard_flag', False)
    flags_pct = row.get('pct', 0)

    if discard_flag:
        return 'DISCARDED'
    
    # Auto-boost for explicit title match
    if 'software' in title:
        prob = max(prob, 0.9)
        
    if prob < SOFTWARE_THRESHOLD:
        return 'DISCARDED'
    
    if flags_pct >= 85:
        return 'HIGH'
    elif flags_pct >= 65:
        return 'MEDIUM'
    else:
        return 'LOW'

def reasons_for_row(r):
    title = str(r.get('Title', '')).lower()
    reasons = []

    st = r.get('state_abbr', '')
    tier = STATE_TIERS.get(st, None)
    if tier:
        reasons.append(f"Tier {tier} state")
    else:
        reasons.append("Tier unknown")

    if r.get('discard_flag', False):
        reasons.append("DISCARDED: Location contains forbidden term")
        return "; ".join(reasons)

    if not r.get('due_date_ok', True):
        reasons.append("Due date too soon (< 7 days)")

    if 'software' in title:
        reasons.append("Title contains 'software' (auto-boosted)")

    # Check raw prob against threshold if 'software' not in title
    if r['software_prob'] < SOFTWARE_THRESHOLD and 'software' not in title:
        reasons.append(f"DISCARDED: Not software (prob: {r['software_prob']:.2f})")
        return "; ".join(reasons)

    if r['pct'] >= 85:
        reasons.append("Strong match to software-related terms")
    elif r['pct'] >= 65:
        reasons.append("Moderate match to software-related terms")
    else:
        reasons.append("Weak match to software-related terms")

    return "; ".join(reasons)