# Mini-ATS Implementation Summary

## What Was Built

A complete single-file Streamlit application (`streamlit_app.py`) that transforms the original JD2GH candidate finder into a full-featured mini Applicant Tracking System (ATS).

## File Structure

```
/Users/amirsh/Documents/Repo/hack/
├── streamlit_app.py          # Main application (~1800 lines, single file)
├── requirements.txt           # Python dependencies
├── .env                       # API keys (gitignored)
├── .env.example              # Environment template
├── README.md                 # Complete documentation
├── run_streamlit.sh          # Startup script
├── ats.db                    # SQLite database (auto-created)
└── venv/                     # Python virtual environment
```

## Key Features Implemented

### 1. Database Layer (SQLModel)

**7 Models with Timestamps:**
- `JobPosting` - Job details, parsed requirements, weights, stats
- `Candidate` - GitHub profiles, portfolios
- `JobCandidateMatch` - Per-job scoring, evidence, status
- `Invitation` - Tokenized invite links with expiry
- `AssessmentTemplate` - Test questions (Soft/Tech)
- `AssessmentAttempt` - Results, duration, anti-cheat metrics

**Database Functions:**
- `init_db()` - Auto-creates tables on startup
- `recalc_job_stats()` - Updates candidate counts
- Automatic timestamps (created_at, updated_at)

### 2. Utilities

**File Upload Support:**
- `extract_text_from_upload()` - Supports PDF, DOCX, MD, TXT
- Uses pypdf for PDF extraction
- Uses python-docx for Word documents

**Token Management:**
- `make_token()` - URL-safe invitation tokens
- `parse_token()` - Token validation and parsing

**JSON Helpers:**
- `json_dumps()` - Safe serialization
- `json_loads()` - Safe deserialization with defaults

### 3. AI & GitHub Integration

**Gemini Wrapper:**
- `gemini_extract_spec()` - Extracts structured JD data
- Returns: role, languages, topics, must_have, nice_to_have
- Lowercase normalization for matching

**GitHub Discovery:**
- `run_discovery_for_job()` - Reuses existing search_github_users()
- Scores candidates using existing score_user()
- Returns: login, name, location, followers, stars, portfolio, scores, evidence

**Assessment Templates:**
- `seed_assessment_templates()` - Creates default tests
- Soft Skills: 5 MCQs (7 minutes)
- Technical: 8 MCQs (20 minutes)

### 4. Multi-Page Navigation

**5 Pages:**

#### Dashboard
- Job posts, candidates, applied, tested metrics
- Active job listings table
- Click-through to other pages

#### Job Postings
- **New Job Tab:**
  - Form: title, city, synonyms, min repos
  - JD input: text area OR file upload
  - AI extraction with Gemini
  - Editable multiselect chips (languages, topics, must_have, nice_to_have)
  - Weight sliders (skills, activity, quality, completeness)
  - Save to database
  
- **Manage Tab:**
  - Card grid of existing jobs
  - Stats per job
  - Edit and delete buttons
  - "Open Dataset" → navigates to Candidates

#### Candidates
- Job selector dropdown
- Config display (city, synonyms, min repos, skills)
- **Run/Refresh Discovery** button
  - Searches GitHub
  - Scores candidates
  - Upserts to database
- **Filters:**
  - "Has all must-haves" checkbox
  - "Active in last 90 days" checkbox
- **Top N Display:**
  - Slider to select top candidates
  - Profile cards with scores and evidence
  - **Invite button** → generates token link
- **Full Dataset Table**
- **Export:** CSV and JSON download buttons

#### Tests (HR)
- Job selector
- Table of all assessment attempts
- Columns: login, name, kind, scores, duration, anti-cheat metrics, status
- Real-time view of candidate test results

#### Candidate Portal
- **Token-based access** via URL parameter
- **Profile Form:**
  - Name, email, LinkedIn, years experience
  - Save → updates candidate record
  - Marks invitation as used
  - Updates match status to "APPLIED"
- **Assessment Buttons:**
  - Start Soft Skills Test (7 min)
  - Start Technical Test (20 min)
  - Shows completion status and scores
- **Active Assessment:**
  - Timer display (MM:SS countdown)
  - Auto-submit on timeout
  - Anti-cheat tracking display
  - MCQ questions with radio buttons
  - Submit button
  - Scoring with penalties
  - Updates match status to "TESTED"

### 5. Assessment System

**Features:**
- Timer with countdown display
- Auto-submit on timeout
- MCQ questions rendered from templates
- Lightweight anti-cheat JavaScript:
  - Tab switch detection (visibilitychange event)
  - Copy/paste blocking
- Scoring:
  - Soft: 2 points per question, max 10
  - Tech: Scaled to 10 points
  - Penalties: -1 for >2 tab switches, -1 for copy/paste
- Results stored in AssessmentAttempt table
- Job stats auto-updated

### 6. Reused Code

**From Original App:**
- `extract_jd_spec()` - Gemini extraction (kept as fallback)
- `search_github_users()` - GitHub GraphQL search (async)
- `score_user()` - Scoring algorithm with subscores
- All existing GitHub query logic
- All existing scoring formulas

**New Wrappers:**
- `gemini_extract_spec()` - Calls original, adds normalization
- `run_discovery_for_job()` - Orchestrates search + scoring

## Technical Constraints Met

✅ **Single file** - Everything in streamlit_app.py  
✅ **Reuse existing code** - All GitHub/Gemini functions preserved  
✅ **SQLite + SQLModel** - Local persistence with ORM  
✅ **Minimal dependencies** - Only added pypdf, python-docx  
✅ **No email** - Copyable invitation links  
✅ **Lightweight anti-cheat** - JavaScript event listeners  
✅ **Simple UI** - Clean Streamlit components, no custom CSS  

## Dependencies Added

```
pypdf          # PDF text extraction
python-docx    # DOCX text extraction
```

(All others were already in requirements.txt)

## Testing Checklist

- [x] Database initialization on startup
- [x] Job creation with JD extraction
- [x] File upload (PDF, DOCX, MD, TXT)
- [x] Editable requirement chips
- [x] Weight sliders
- [x] GitHub discovery and scoring
- [x] Candidate filtering
- [x] Invitation generation
- [x] Token-based portal access
- [x] Profile form save
- [x] Assessment timer
- [x] MCQ rendering
- [x] Anti-cheat tracking
- [x] Assessment submission and scoring
- [x] Dashboard metrics
- [x] CSV/JSON export

## Usage Flow

1. **Start:** `./run_streamlit.sh` or `streamlit run streamlit_app.py`
2. **Create Job:** Job Postings → New Job → upload/paste JD → extract → edit → save
3. **Find Candidates:** Candidates → select job → Run Discovery
4. **Invite:** Top N → click Invite → copy link
5. **Candidate Flow:** Opens link → fills profile → takes tests
6. **Review:** Tests (HR) → view scores and anti-cheat data
7. **Track:** Dashboard → see overall metrics

## Code Statistics

- **Total Lines:** ~1800
- **Models:** 7 SQLModel classes
- **Pages:** 5 Streamlit pages
- **Functions:** ~15 utility/wrapper functions
- **Reused:** 3 original functions (extract_jd_spec, search_github_users, score_user)
- **Database Tables:** 6 tables with auto-timestamps
- **Assessment Questions:** 13 total (5 soft + 8 tech)

## What Makes This a "Mini-ATS"

1. **Complete Lifecycle:** Discovery → Invitation → Application → Assessment → Review
2. **Database Persistence:** All data stored locally with relationships
3. **Multi-User Support:** Separate candidate portal vs HR views
4. **Assessment Platform:** Timed tests with scoring and anti-cheat
5. **Analytics:** Dashboard with KPIs and job stats
6. **Export:** CSV/JSON for external analysis

## Future Enhancements (Not Implemented)

- Email integration (SendGrid, AWS SES)
- PostgreSQL for multi-user/production
- Enterprise proctoring APIs
- Advanced analytics dashboard
- Video interviews
- ATS integrations (Greenhouse, Lever)
- Candidate pipeline stages
- Automated email campaigns
