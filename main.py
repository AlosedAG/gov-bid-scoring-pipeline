!pip install -q transformers datasets scikit-learn pandas joblib torch
!pip install -q rapidfuzz

import pandas as pd
import numpy as np
import joblib
import torch
import re, requests
import google.auth
import gspread

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from transformers.trainer import Trainer
from datasets import Dataset
from urllib.parse import urlparse
from datetime import datetime, timedelta, UTC
from transformers import pipeline
from google.colab import auth, drive
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from rapidfuzz import process, fuzz
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

import pickle

model_path = "MyDrive/your_folder/bid_software_model.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

auth.authenticate_user()
drive.mount('/content/drive')

creds, _ = google.auth.default()
gc = gspread.authorize(creds)

bidOp = "______"
sh = gc.open(____)
print("Opened spreadsheet:", _____)

# drive.mount('/content/drive', force_remount=True) - in case of error mounting drive

try:
    ws_csv = sh.worksheet("CSV")
    csv_raw = get_as_dataframe(ws_csv, evaluate_formulas=True).dropna(how="all")
    print(f"Found CSV tab with {len(csv_raw)} rows")

    csv_raw.columns = [f"Col_{i}" for i in range(len(csv_raw.columns))]

    bids_df = pd.DataFrame({
        'Location': csv_raw['Col_31'].astype(str) + ', ' + csv_raw['Col_4'].astype(str),  # AF + E
        'Due Date': csv_raw['Col_45'],
        'Link': csv_raw['Col_48'],
        'Title': csv_raw['Col_49'],
        'Phone': csv_raw['Col_36']
    })

    def fix_date_format(date_str):
        if pd.isna(date_str) or date_str == '':
            return ''
        date_str = str(date_str)
        if 'T' in date_str:
            return date_str.split('T')[0]
        return date_str[:10]

    bids_df['Due Date'] = bids_df['Due Date'].apply(fix_date_format)
    bids_df['ID'] = range(1, len(bids_df) + 1)
    bids_df['Description'] = ''
    bids_df = bids_df[['ID', 'Location', 'Title', 'Link', 'Due Date', 'Description', 'Phone']]
    bids_df = bids_df.dropna(subset=['Location', 'Title', 'Link'], how='all')

    print(f"Processed {len(bids_df)} bids from CSV")

except Exception as e:
    print(f"Error processing CSV tab: {e}")
    raise

# Read populations tab
ws_populations = sh.worksheet("populations")
pops_df = get_as_dataframe(ws_populations, evaluate_formulas=True).dropna(how="all").fillna('')

# Read existing landscapes tab
try:
    ws_landscapes = sh.worksheet("existing_landscapes")
    landscapes_df = get_as_dataframe(ws_landscapes, evaluate_formulas=True).dropna(how="all").fillna('')
    landscapes_df.columns = ['Landscape_Title', 'Landscape_Description']
    landscapes_df = landscapes_df[landscapes_df['Landscape_Title'].astype(str).str.strip() != '']
    print(f"Loaded {len(landscapes_df)} landscapes")
except Exception as e:
    print(f"Warning: Could not load existing_landscapes tab: {e}")
    landscapes_df = pd.DataFrame(columns=['Landscape_Title', 'Landscape_Description'])

print("bid rows:", len(bids_df), "pops rows:", len(pops_df))

# --- Constants and helper functions ---

STATE_ABBR = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO",
    "Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID",
    "Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA",
    "Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS",
    "Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ",
    "New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK",
    "Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD",
    "Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA",
    "West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY","District of Columbia":"DC"
}

ABBR_TO_STATE = {v: k for k, v in STATE_ABBR.items()}

STATE_TIERS = {
    'AZ': 1, 'CA': 1, 'FL': 1, 'GA': 1, 'IL': 1, 'IN': 1,
    'MI': 1, 'NV': 1, 'NY': 1, 'NC': 1, 'PA': 1, 'TX': 1,
    'VA': 1, 'WI': 1,
    'CO': 2, 'CT': 2, 'IA': 2, 'LA': 2, 'MD': 2, 'MA': 2,
    'MN': 2, 'MO': 2, 'NH': 2, 'NJ': 2, 'OH': 2, 'OR': 2,
    'SC': 2, 'TN': 2, 'UT': 2, 'WA': 2,
    'AL': 3, 'AK': 3, 'AR': 3, 'DE': 3, 'HI': 3, 'ID': 3,
    'KS': 3, 'KY': 3, 'ME': 3, 'MS': 3, 'MT': 3, 'NE': 3,
    'NM': 3, 'ND': 3, 'OK': 3, 'RI': 3, 'SD': 3, 'VT': 3,
    'WV': 3, 'WY': 3
}

DISCARD_TERMS = [
    "state",
    "transport",
    "airport",
    "town",
    "library",
    "department",
    "school",
    "water",
    "office",
    "commerce",
    "utilities",
    "agency",
    "sanitation",
    "district",
    "council",
    "clinical",
    "human services",
    "waste",
    "health",
]

# RFx terms to DISCARD (non-RFP solicitation types)
RFX_TERMS = [
    'rfi', 'request for information',
    'rfq', 'request for quotation', 'request for quote',
    'rfb', 'request for bid',
    'ifb', 'invitation for bid',
    'itb', 'invitation to bid',
    'rfs', 'request for services',
    'rft', 'request for tender',
    'roi', 'registration of interest',
    'eoi', 'expression of interest',
    'soq', 'statement of qualifications',
    'ssq', 'supplier statement of qualifications',
    'rfsq', 'request for supplier qualifications',
    'rfqual', 'request for qualifications',
    'rfiq', 'request for information and qualifications',
    'pqq', 'pre-qualification questionnaire',
    'cn', 'contract notice',
    'pin', 'prior information notice',
    'dni', 'draft notice for information',
    'bafo', 'best and final offer',
    'loi', 'letter of intent',
    'bpa', 'blanket purchase agreement',
    'idiq', 'indefinite delivery indefinite quantity',
    'matoc', 'multiple award task order contract',
    'satoc', 'single award task order contract'
]

AGG_DOMAINS = [
    'govspend','rfpdb','findrfp','openminds','bidsearch','bidprime','therfpdatabase',
    'bidnet','governmentbids.com','bidbanana','rfpmart','govwin','demandstar',
    'govtribe','mygovwatch','bidsusa','governmentbids'
]

# ============================================================================
# FIXED v3: More carefully curated SOFTWARE_KEYWORDS
# ============================================================================

# PRIMARY keywords - STRONG software indicators
# These are specific enough that they almost always mean software/IT
SOFTWARE_KEYWORDS_PRIMARY = [
    # Explicit software terms
    "software",
    "saas",
    "paas",
    "iaas",

    # IT Platform terms (but NOT just "system" alone - too generic)
    "platform",  # Added - "communication platform" = software
    "application",  # software application

    # Specific software types
    "erp",
    "crm",
    "hris",
    "cms",
    "lms",

    # Security software
    "mdr",  # Managed Detection and Response
    "xdr",  # Extended Detection and Response
    "edr",  # Endpoint Detection and Response
    "siem",  # Security Information and Event Management
    "soar",  # Security Orchestration, Automation and Response

    # Database/data specific
    "database",
    "sql",
    "nosql",
    "data warehouse",
    "etl",

    # Web/mobile development
    "web application",
    "mobile app",
    "mobile application",
    "app development",
    "web portal",
    "citizen portal",
    "website modernization",
    "website redesign",
    "website development",

    # IT Infrastructure (specific)
    "cloud migration",
    "cloud hosting",
    "cloud services",
    "cloud platform",
    "api gateway",
    "api integration",

    # Cybersecurity (specific)
    "cybersecurity",
    "cyber security",

    # AI/ML specific
    "machine learning",
    "artificial intelligence",

    # GIS specific
    "gis",
    "arcgis",
    "esri",
    "geospatial",

    # Specific software categories
    "enterprise resource planning",
    "customer relationship management",
    "business intelligence",
    "permitting software",
    "licensing software",
    "case management software",
    "document management software",
    "records management software",
    "billing software",
    "accounting software",
    "payroll software",
    "timekeeping software",
    "scheduling software",
    "fleet management software",
    "asset management software",
    "inventory software",

    # Communication platforms (software)
    "communication platform",
    "mass communication",
    "mass notification",
    "unified communications",
]

# SECONDARY keywords - only boost if ML probability is already decent (>= 0.50)
SOFTWARE_KEYWORDS_SECONDARY = [
    "implementation",
    "integration",
    "digital transformation",
    "e-government",
    "smart city",
    "automation",
    "workflow",
    "analytics",
    "dashboard",
    "reporting tool",
    "modernization",
]

# ML probability thresholds
SOFTWARE_THRESHOLD_HIGH = 0.70  # Strong ML signal
SOFTWARE_THRESHOLD_MEDIUM = 0.50  # Decent ML signal (can be boosted by keywords)

CITY_POP_THRESHOLD = 75000
COUNTY_POP_THRESHOLD = 150000
DUE_DATE_DAYS_MIN = 7
LANDSCAPE_MATCH_THRESHOLD = 70


def compile_rfx_patterns():
    def escape_regex(s: str) -> str:
        return re.escape(s)

    patterns = []
    for term in RFX_TERMS:
        t = term.lower()
        if re.match(r"^[a-z]{3,5}$", t, re.IGNORECASE):
            letters = r"[\s\-._]?".join(escape_regex(c) for c in t)
            patterns.append(re.compile(rf"\b{letters}\b", re.IGNORECASE))
            patterns.append(re.compile(rf"\b{letters}[\s:\-–—)?]?", re.IGNORECASE))
            patterns.append(re.compile(rf"\(\s*{letters}\s*\)", re.IGNORECASE))
            continue
        flexible = r"[\s\-_/.,]+".join(escape_regex(word) for word in t.split())
        patterns.append(re.compile(rf"\b{flexible}\b", re.IGNORECASE))
        patterns.append(re.compile(rf"\b{flexible}s?\b", re.IGNORECASE))
    return patterns

COMPILED_RFX_PATTERNS = compile_rfx_patterns()
DISCARD_TERMS_REGEX = re.compile("|".join(DISCARD_TERMS), re.I)
CITY_OF_REGEX = re.compile(r"^(city of|county of)\s+", re.I)
COUNTY_WORD_REGEX = re.compile(r"\bcounty\b", re.I)
CITY_WORD_REGEX = re.compile(r"\bcity\b", re.I)
GOV_CITY_REGEX = re.compile(r"\b(the city of|city of|city)\b", re.I)
GOV_COUNTY_REGEX = re.compile(r"\b(the county of|county of|county)\b", re.I)
RFX_REGEX = re.compile(r'\b(' + '|'.join(re.escape(p) for p in RFX_TERMS) + r')\b', re.I)

def parse_location_field(loc):
    if not loc or pd.isna(loc):
        return ('', False, '', False)

    s = str(loc).strip().lower()
    discard_flag = bool(DISCARD_TERMS_REGEX.search(s))
    is_county = 'county' in s

    s = CITY_OF_REGEX.sub('', s).strip()

    if ',' in s:
        name, state = [p.strip() for p in s.rsplit(',', 1)]
    else:
        parts = s.split()
        if len(parts) > 1:
            name = ' '.join(parts[:-1])
            state = parts[-1].strip()
        else:
            name = s
            state = ''

    name = COUNTY_WORD_REGEX.sub('', name).strip()

    if 'city' in name.lower() and not name.lower().endswith('city'):
        name = CITY_WORD_REGEX.sub('', name).strip()
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

for df in [bids_df, pops_df]:
    parsed = df['Location'].astype(str).apply(parse_location_field)
    df[['place_name', 'is_county', 'state_abbr', 'discard_flag']] = pd.DataFrame(parsed.tolist(), index=df.index)
    df['normalized_location'] = df.apply(
        lambda r: format_location_name(r['place_name'], r['is_county'], r['state_abbr']),
        axis=1
    )

pops_df['place_name'] = pops_df['Location'].apply(lambda loc: parse_location_field(loc)[0])
pops_df['state_abbr'] = pops_df['Location'].apply(lambda loc: parse_location_field(loc)[2])

merged = bids_df.merge(
    pops_df[['place_name','state_abbr','Population']],
    how='left',
    left_on=['place_name', 'state_abbr'],
    right_on=['place_name', 'state_abbr'],
    suffixes=('','_pop')
)

merged['is_county_or_city'] = (merged['is_county']) | (merged['place_name'].fillna('').astype(str).str.lower().str.contains('city'))

def pop_ok(row):
    pop = row.get('Population')
    if pd.isna(pop) or pop=='':
        return False
    try:
        pop = int(pop)
    except:
        return False
    if row.get('is_county'):
        return pop >= COUNTY_POP_THRESHOLD
    else:
        return pop >= CITY_POP_THRESHOLD

merged['pop_ok'] = merged.apply(pop_ok, axis=1)


def is_aggregator(link):
    if pd.isna(link) or link == '':
        return False
    link_lower = str(link).lower()
    return any(domain in link_lower for domain in AGG_DOMAINS)

merged['is_aggregator'] = merged['Link'].apply(is_aggregator)
merged['not_aggregator'] = ~merged['is_aggregator']

def check_rfx(title, description):
    text = f"{title} {description}".lower()
    return any(pattern.search(text) for pattern in COMPILED_RFX_PATTERNS)

merged['is_RFx'] = merged.apply(lambda r: check_rfx(r['Title'], r['Description']), axis=1)
merged['not_RFx'] = ~merged['is_RFx']

def is_gov_related(row):
    desc = str(row.get('Description','')).lower()
    link = str(row.get('Link','')).lower()
    if GOV_CITY_REGEX.search(desc) or GOV_COUNTY_REGEX.search(desc):
        return True
    try:
        dom = urlparse(link).netloc
        if dom.endswith('.gov'):
            return True
    except:
        pass
    return False

merged['gov_related'] = merged.apply(is_gov_related, axis=1)

today = datetime.now(UTC)
merged['Due Date_parsed'] = pd.to_datetime(merged['Due Date'], errors='coerce').dt.tz_localize(UTC)
merged['due_date_ok'] = merged['Due Date_parsed'] >= (today + timedelta(days=DUE_DATE_DAYS_MIN))

merged['text'] = (merged['Title'].fillna('').astype(str) + '||' + merged['Description'].fillna('').astype(str)).str.strip()

model_pipeline = None
try:
    model_pipeline = joblib.load("/content/drive/MyDrive/bid_software_model.pkl")
except FileNotFoundError:
    print("Error: The model file 'bid_software_model.pkl' was not found. Please ensure it exists at the specified path.")
except Exception as e:
    print(f"An unexpected error occurred while loading the model: {e}")


# ============================================================================
# FIXED v3: Keyword detection functions
# ============================================================================
def keyword_boost_primary(text):
    """
    Check if text contains PRIMARY software-related keywords.
    These are strong indicators that should boost ranking.
    """
    if pd.isna(text) or text == "":
        return False
    text_lower = str(text).lower()
    return any(keyword.lower() in text_lower for keyword in SOFTWARE_KEYWORDS_PRIMARY)


def keyword_boost_secondary(text):
    """
    Check if text contains SECONDARY software-related keywords.
    These alone are not enough - need ML probability support.
    """
    if pd.isna(text) or text == "":
        return False
    text_lower = str(text).lower()
    return any(keyword.lower() in text_lower for keyword in SOFTWARE_KEYWORDS_SECONDARY)


# Apply keyword detection
merged["has_primary_keywords"] = merged["text"].apply(keyword_boost_primary)
merged["has_secondary_keywords"] = merged["text"].apply(keyword_boost_secondary)
merged["has_software_keywords"] = merged["has_primary_keywords"]


def get_software_probabilities(texts, model_pipeline):
    if model_pipeline is None:
        print("Warning: Model not loaded. Returning default probability of 0.0 for all texts.")
        return [0.0] * len(texts)

    valid_texts = [str(t) if t and not pd.isna(t) else "" for t in texts]

    try:
        preds = model_pipeline.predict_proba(valid_texts)
        return [p[1] for p in preds]
    except Exception as e:
        print(f"CRITICAL ERROR: Model prediction failed: {str(e)}. Returning default probabilities.")
        return [0.0] * len(texts)


merged['software_prob'] = get_software_probabilities(merged['text'].tolist(), model_pipeline)
merged['software_model_match'] = merged['software_prob'] >= SOFTWARE_THRESHOLD_HIGH

weight_flags = ['is_county_or_city', 'pop_ok', 'not_aggregator', 'not_RFx', 'gov_related', 'due_date_ok']
merged['flags_score'] = merged[weight_flags].sum(axis=1).astype(float)
merged['pct'] = (merged['flags_score'] / len(weight_flags)) * 100


# ============================================================================
# FIXED v3: assign_rank function
# ============================================================================
def assign_rank(row):
    """
    Assign rank based on:
    1. "software" explicitly in title -> HIGH
    2. PRIMARY keywords -> HIGH/MEDIUM (strong software signal)
    3. ML prob >= 0.70 -> rank based on prob strength
    4. ML prob >= 0.50 + SECONDARY keywords -> MEDIUM (boosted)
    5. Otherwise -> DISCARDED

    Key changes in v3:
    - ML prob >= 0.70 gets at least MEDIUM (was getting LOW before)
    - ML prob >= 0.80 gets HIGH regardless of flags
    - "platform", "mdr", "xdr" added to PRIMARY keywords
    - Removed generic "system" from keywords
    """
    title = str(row.get("Title", "")).lower()
    prob = row.get("software_prob", 0)
    discard_flag = row.get("discard_flag", False)
    flags_pct = row.get("pct", 0)
    has_primary_keywords = row.get("has_primary_keywords", False)
    has_secondary_keywords = row.get("has_secondary_keywords", False)

    has_software_in_title = "software" in title

    # Override discard_flag for strong software signals
    if has_software_in_title or has_primary_keywords or prob >= SOFTWARE_THRESHOLD_HIGH:
        discard_flag = False

    if discard_flag:
        return "DISCARDED"

    # =========================================================================
    # RANKING LOGIC v3
    # =========================================================================

    # CASE 1: "software" explicitly in title -> HIGH
    if has_software_in_title:
        return "HIGH"

    # CASE 2: Has PRIMARY software keywords -> HIGH or MEDIUM
    if has_primary_keywords:
        if prob >= SOFTWARE_THRESHOLD_MEDIUM or flags_pct >= 50:
            return "HIGH"
        else:
            return "MEDIUM"

    # CASE 3: Strong ML signal (prob >= 0.80) -> HIGH regardless of flags
    if prob >= 0.80:
        return "HIGH"

    # CASE 4: Good ML signal (prob >= 0.70) -> at least MEDIUM
    if prob >= SOFTWARE_THRESHOLD_HIGH:
        if flags_pct >= 65:
            return "HIGH"
        else:
            return "MEDIUM"  # Changed from LOW - 0.70+ ML should be at least MEDIUM

    # CASE 5: Decent ML signal (prob >= 0.50) + SECONDARY keywords -> MEDIUM
    if prob >= SOFTWARE_THRESHOLD_MEDIUM and has_secondary_keywords:
        return "MEDIUM"

    # CASE 6: Not software - discard
    return "DISCARDED"


def get_tier(row):
    return STATE_TIERS.get(row.get("state_abbr", ""), None)


def get_reasons(row):
    title = str(row.get("Title", "")).lower()
    reasons = []

    has_software_in_title = "software" in title
    has_primary_keywords = row.get("has_primary_keywords", False)
    has_secondary_keywords = row.get("has_secondary_keywords", False)
    prob = row["software_prob"]

    tier = STATE_TIERS.get(row.get("state_abbr", ""), None)
    if tier:
        reasons.append(f"Tier {tier} state")
    else:
        reasons.append("Tier unknown")

    if row.get("discard_flag", False) and not has_software_in_title and not has_primary_keywords and prob < SOFTWARE_THRESHOLD_HIGH:
        reasons.append("DISCARDED: Location contains forbidden term")
        return "; ".join(reasons)

    if not row.get("due_date_ok", True):
        reasons.append("Due date too soon (< 7 days)")

    if has_software_in_title:
        reasons.append("Title contains 'software' (auto-HIGH)")
    elif has_primary_keywords:
        reasons.append("Contains primary software keywords (HIGH/MEDIUM)")
    elif has_secondary_keywords and prob >= SOFTWARE_THRESHOLD_MEDIUM:
        reasons.append("Contains secondary keywords + ML support (MEDIUM)")

    if prob >= 0.80:
        reasons.append(f"Strong ML signal (prob: {prob:.2f}) -> HIGH")
    elif prob >= SOFTWARE_THRESHOLD_HIGH:
        reasons.append(f"Good ML signal (prob: {prob:.2f}) -> MEDIUM+")
    elif prob >= SOFTWARE_THRESHOLD_MEDIUM:
        reasons.append(f"Decent ML signal (prob: {prob:.2f})")
    else:
        if not has_software_in_title and not has_primary_keywords:
            reasons.append(f"DISCARDED: Not software (prob: {prob:.2f})")
            return "; ".join(reasons)

    pct = row["pct"]
    if pct >= 85:
        reasons.append("Strong flags match")
    elif pct >= 65:
        reasons.append("Moderate flags match")
    else:
        reasons.append("Weak flags match")

    return "; ".join(reasons)


def generate_enhanced_description(row):
    place_name = row.get("place_name", "")
    is_county = row.get("is_county", False)
    state_abbr = row.get("state_abbr", "")
    title = str(row.get("Title", ""))
    title_lower = title.lower()

    state_full = ABBR_TO_STATE.get(state_abbr, state_abbr)

    if is_county:
        location_prefix = f"The County of {place_name}, {state_full}"
    else:
        location_prefix = f"The City of {place_name}, {state_full}"

    action_verbs = []

    if any(kw in title_lower for kw in ["cloud", "migration", "implementation", "arcgis", "gis"]):
        action_verbs = [
            "is seeking to implement",
            "has issued a procurement for",
            "is modernizing its infrastructure with",
            "requires implementation services for",
        ]
    elif any(kw in title_lower for kw in ["software", "system", "platform", "application", "erp", "crm"]):
        action_verbs = [
            "is procuring",
            "seeks qualified vendors for",
            "has published an RFP for",
            "is requesting proposals for",
        ]
    elif any(kw in title_lower for kw in ["consulting", "professional services", "advisory"]):
        action_verbs = [
            "requires professional expertise in",
            "is seeking consulting services for",
            "seeks qualified professionals for",
            "has issued a request for professional services related to",
        ]
    elif any(kw in title_lower for kw in ["design", "engineering", "architecture"]):
        action_verbs = [
            "is soliciting design proposals for",
            "seeks engineering services for",
            "requires architectural expertise in",
            "has issued an RFP for design services related to",
        ]
    elif any(kw in title_lower for kw in ["maintenance", "support", "managed services"]):
        action_verbs = [
            "is procuring maintenance services for",
            "seeks ongoing support for",
            "requires managed services related to",
            "is requesting proposals for the maintenance of",
        ]
    elif any(kw in title_lower for kw in ["hardware", "equipment", "camera", "kiosk", "x-ray", "scanner", "lidar", "device", "sensor"]):
        action_verbs = [
            "is seeking to acquire",
            "has issued a procurement for",
            "is requesting proposals for the purchase of",
            "seeks qualified vendors for",
        ]
    elif any(kw in title_lower for kw in ["renewal", "support", "maintenance", "subscription", "hosting"]):
        action_verbs = [
            "requires renewal and support services for",
            "is seeking maintenance services for",
            "has issued an RFP for ongoing support of",
            "is procuring subscription services for",
        ]
    elif any(kw in title_lower for kw in ["data", "analytics", "reporting", "telematics", "monitoring", "dashboard"]):
        action_verbs = [
            "is seeking data management solutions for",
            "requires analytics capabilities through",
            "has published an RFP for reporting services related to",
            "is procuring data solutions for",
        ]
    elif any(kw in title_lower for kw in ["cybersecurity", "security", "mdr", "xdr", "edr", "detection", "falcon"]):
        action_verbs = [
            "is strengthening security infrastructure with",
            "requires cybersecurity solutions for",
            "has issued an RFP for security services related to",
            "is procuring threat detection capabilities through",
        ]
    elif any(kw in title_lower for kw in ["record", "document", "archive", "foia", "evidence", "digital evidence"]):
        action_verbs = [
            "is modernizing records management with",
            "seeks document management solutions for",
            "has issued an RFP for digital records related to",
            "requires records management services for",
        ]
    elif any(kw in title_lower for kw in ["payment", "billing", "collections", "financial", "accounting", "erp"]):
        action_verbs = [
            "is modernizing financial operations with",
            "seeks payment processing solutions for",
            "has published an RFP for billing systems related to",
            "requires financial management software for",
        ]
    elif any(kw in title_lower for kw in ["emergency", "ems", "cad", "dispatch", "911", "public safety"]):
        action_verbs = [
            "is enhancing emergency response capabilities with",
            "requires public safety systems for",
            "has issued an RFP for emergency services related to",
            "seeks mission-critical solutions for",
        ]
    elif any(kw in title_lower for kw in ["case management", "workflow", "permitting", "inspection", "licensing"]):
        action_verbs = [
            "is streamlining operations with",
            "seeks case management solutions for",
            "has published an RFP for workflow automation related to",
            "requires process management software for",
        ]
    elif any(kw in title_lower for kw in ["website", "web", "portal", "online", "digital", "mobile app"]):
        action_verbs = [
            "is enhancing digital presence with",
            "seeks web development services for",
            "has issued an RFP for online services related to",
            "requires digital platform development for",
        ]
    elif any(kw in title_lower for kw in ["communication", "telephony", "messaging", "email", "social media"]):
        action_verbs = [
            "is upgrading communication infrastructure with",
            "seeks unified communication solutions for",
            "has published an RFP for telephony services related to",
            "requires collaboration tools for",
        ]
    elif any(kw in title_lower for kw in ["cloud", "migration", "hosting", "server", "infrastructure"]):
        action_verbs = [
            "is modernizing IT infrastructure with",
            "seeks cloud migration services for",
            "has issued an RFP for infrastructure solutions related to",
            "requires cloud hosting services for",
        ]
    elif any(kw in title_lower for kw in ["gis", "mapping", "arcgis", "esri", "geospatial"]):
        action_verbs = [
            "is advancing geospatial capabilities with",
            "seeks GIS implementation services for",
            "has published an RFP for mapping solutions related to",
            "requires geographic information systems for",
        ]
    elif any(kw in title_lower for kw in ["hris", "hr", "timekeeping", "payroll", "workforce", "employee"]):
        action_verbs = [
            "is modernizing workforce management with",
            "seeks human resources solutions for",
            "has issued an RFP for HRIS systems related to",
            "requires employee management software for",
        ]
    elif any(kw in title_lower for kw in ["library", "ils", "discovery", "education", "learning"]):
        action_verbs = [
            "is enhancing educational services with",
            "seeks library management solutions for",
            "has published an RFP for educational technology related to",
            "requires learning management systems for",
        ]
    elif any(kw in title_lower for kw in ["fleet", "asset", "vehicle", "equipment management", "inventory"]):
        action_verbs = [
            "is optimizing asset operations with",
            "seeks fleet management solutions for",
            "has issued an RFP for asset tracking systems related to",
            "requires inventory management software for",
        ]
    elif any(kw in title_lower for kw in ["ai", "artificial intelligence", "machine learning", "ml"]):
        action_verbs = [
            "is exploring AI capabilities with",
            "seeks artificial intelligence solutions for",
            "has published an RFP for machine learning services related to",
            "requires AI-powered systems for",
        ]
    elif any(kw in title_lower for kw in ["management", "administration", "oversight"]):
        action_verbs = [
            "requires management services for",
            "seeks administrative solutions for",
            "has issued an RFP for oversight services related to",
            "is procuring management capabilities for",
        ]
    else:
        action_verbs = [
            "is seeking proposals for",
            "has issued an RFP for",
            "is accepting bids for",
            "requests qualifications for",
            "is soliciting vendors for",
        ]

    hash_value = hash(place_name + state_abbr + title)
    selected_action = action_verbs[abs(hash_value) % len(action_verbs)]

    description = f"{location_prefix} {selected_action} {title}"

    if "sole source" in title_lower:
        description += ". This is a sole source procurement"
    elif "cloud" in title_lower and "gis" in title_lower:
        description += ". This initiative involves transitioning to cloud-based geographic information systems"
    elif "digital transformation" in title_lower:
        description += ". This project aims to modernize municipal operations through technology"
    elif "cybersecurity" in title_lower or "security" in title_lower or "mdr" in title_lower or "xdr" in title_lower:
        description += ". This procurement focuses on enhancing information security infrastructure"
    elif "citizen portal" in title_lower or "online services" in title_lower:
        description += ". This project will improve digital access to government services"
    elif "emergency" in title_lower or "public safety" in title_lower:
        description += ". This system is critical for public safety operations"
    elif "cad" in title_lower or "dispatch" in title_lower:
        description += ". This solution supports emergency response coordination"
    elif "erp" in title_lower or "financial" in title_lower:
        description += ". This system will streamline financial operation and reporting"

    return description


merged['Rank'] = merged.apply(assign_rank, axis=1)
merged['Tier'] = merged.apply(get_tier, axis=1)
merged['Reasons'] = merged.apply(get_reasons, axis=1)

merged["Description"] = merged.apply(generate_enhanced_description, axis=1)


def match_landscape(title):
    if pd.isna(title) or title == '' or len(landscapes_df) == 0:
        return 'NO'

    title_str = str(title).strip()
    if title_str == '':
        return 'NO'

    landscape_titles = landscapes_df['Landscape_Title'].tolist()
    result = process.extractOne(
        title_str,
        landscape_titles,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=LANDSCAPE_MATCH_THRESHOLD
    )

    if result:
        return result[0]
    else:
        return 'NO'

merged['Landscape'] = merged['Title'].apply(match_landscape)

# Output columns
out_cols = ['ID', 'normalized_location', 'Title', 'Link', 'Due Date', 'Rank', 'pct', 'software_prob', 'has_primary_keywords', 'has_secondary_keywords', 'Reasons', 'Description', 'Phone', 'Landscape']
result_df = merged[out_cols].copy()

# Write to results tab
try:
    ws_results = sh.worksheet("results")
    ws_results.clear()
except:
    ws_results = sh.add_worksheet(title="results", rows="1000", cols="20")
set_with_dataframe(ws_results, result_df.reset_index(drop=True))
print("Wrote results (%d rows) to 'results' sheet." % len(result_df))

print("\nRank distribution:")
print(result_df['Rank'].value_counts())

print("\nLandscape matches:")
print(result_df['Landscape'].value_counts())

# Save to CSV
csv_path = "/content/drive/MyDrive/bid_results.csv"
result_df.to_csv(csv_path, index=False)
print(f"\nSaved results to {csv_path}")

