import re

# --- Column Indices (0-based) based on original logic ---
# Adjust these if your input CSV layout changes
COL_IDX_LOCATION_MAIN = 31 # Originally Col_31
COL_IDX_LOCATION_STATE = 4  # Originally Col_4 (used to combine with main location)
COL_IDX_DUE_DATE = 45       # Originally Col_45
COL_IDX_LINK = 48           # Originally Col_48
COL_IDX_TITLE = 49          # Originally Col_49
COL_IDX_PHONE = 36          # Originally Col_36

# --- Thresholds ---
SOFTWARE_THRESHOLD = 0.58
CITY_POP_THRESH = 75000
COUNTY_POP_THRESH = 150000

# --- State Abbreviations ---
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

# --- State Tiers ---
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

# --- Blocklists and Patterns ---
DISCARD_TERMS = [
    'state', 'transport', 'airport', 'town','library','department','school','water',
    'office','commerce','utilities', 'agency','sanitation','district', 'council', 'clinical',
    'human services', 'waste', 'health'
]

AGG_DOMAINS = [
    'govspend','rfpdb','findrfp','openminds','bidsearch','bidprime','therfpdatabase',
    'bidnet','governmentbids.com','bidbanana','rfpmart','govwin','demandstar',
    'govtribe','mygovwatch','bidsusa','governmentbids'
]

RFX_PATTERNS = [
    'RFI', 'Request for Information', 'RFQ', 'Request for Quotation', 'Request for Quote',
    'RFB', 'Request for Bid', 'IFB', 'Invitation for Bid', 'ITB', 'Invitation to Bid',
    'RFS', 'Request for Services', 'RFT', 'Request for Tender', 'ROI', 'Registration of Interest',
    'EOI', 'Expression of Interest', 'SOQ', 'Statement of Qualifications',
    'SSQ', 'Supplier Statement of Qualifications', 'RFSQ', 'Request for Supplier Qualifications',
    'RFQual', 'Request for Qualifications', 'RFIQ', 'Request for Information and Qualifications',
    'PQQ', 'Pre-Qualification Questionnaire', 'CN', 'Contract Notice', 'PIN', 'Prior Information Notice',
    'DNI', 'Draft Notice for Information', 'BAFO', 'Best and Final Offer', 'LOI', 'Letter of Intent',
    'BPA', 'Blanket Purchase Agreement', 'IDIQ', 'Indefinite Delivery Indefinite Quantity',
    'MATOC', 'Multiple Award Task Order Contract', 'SATOC', 'Single Award Task Order Contract'
]

# Regex compilation
ESCAPED_RFX = [re.escape(p) for p in RFX_PATTERNS]
RFX_REGEX = r'\b(' + '|'.join(ESCAPED_RFX) + r')\b'