# TalentSonar - AI-Powered Applicant Tracking System

![TalentSonar Logo](TalentSonar.png)

An AI-powered ATS that finds GitHub developers matching your job descriptions and manages the entire candidate lifecycle from discovery to assessment.

## Features

### 🎯 Complete ATS Workflow

1. **Dashboard** - Overview metrics and job listings
2. **Job Postings** - Create and manage job postings with AI-powered JD analysis
3. **Candidates** - Discover GitHub developers, score them, and invite top candidates
4. **Tests (HR)** - View assessment results and anti-cheat metrics
5. **Candidate Portal** - Token-based candidate application portal with assessments

### 🤖 AI-Powered Features

- **Gemini AI** extracts structured requirements from job descriptions
- Automatically identifies programming languages, frameworks, and skills
- Generates must-have and nice-to-have requirements

### 🔍 GitHub Discovery

- Searches GitHub users by location and programming languages
- Analyzes repositories, topics, contributions, and activity
- Intelligent scoring algorithm (skills 60%, activity 25%, quality 10%, completeness 5%)

### 📊 Candidate Management

- Automated candidate discovery and scoring
- Customizable filters (must-haves, activity recency)
- Top-N candidate selection
- Invitation system with tokenized links

### �� Assessment System

- **Soft Skills Test** (7 minutes, 5 questions)
- **Technical Test** (20 minutes, 8 questions)
- Anti-cheat features:
  - Tab switch detection
  - Copy/paste prevention
  - Automatic scoring penalties
- Timer with auto-submit

### 💾 Data Persistence

- **PostgreSQL** (production) or SQLite (local dev)
- SQLModel ORM for database operations
- Job postings, candidates, matches, invitations, assessments
- Automatic job statistics tracking
- Connection pooling for production scale

### 📥 Export Options

- CSV and JSON export per job
- Full candidate dataset with scores and evidence

## Installation

### Quick Start (SQLite - Local Development)

1. Clone the repository:
```bash
git clone https://github.com/amirshahdadian/TalentSonar.git
cd TalentSonar
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys:
# - GEMINI_API_KEY (from Google AI Studio)
# - GITHUB_TOKEN (Personal Access Token from GitHub)
# - DATABASE_URL (optional - defaults to SQLite)
```

5. Run the app:
```bash
streamlit run streamlit_app.py
```

### PostgreSQL Setup (Production)

For production deployment with PostgreSQL:

1. **Install PostgreSQL** (see [POSTGRES_SETUP.md](POSTGRES_SETUP.md) for detailed instructions)

2. **Quick setup** (macOS/Linux):
```bash
./setup_postgres.sh
```

3. **Or manually create database**:
```bash
psql postgres
CREATE DATABASE talentsonar;
CREATE USER talentsonar_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE talentsonar TO talentsonar_user;
```

4. **Add to .env**:
```bash
DATABASE_URL=postgresql://talentsonar_user:your_password@localhost:5432/talentsonar
```

5. **Run the app** - it will automatically create all tables!

**Hosted PostgreSQL Options:**
- [Neon](https://neon.tech) - Free tier, serverless
- [Supabase](https://supabase.com) - Free tier with extras
- [Railway.app](https://railway.app) - Easy deployment
- See [POSTGRES_SETUP.md](POSTGRES_SETUP.md) for more options


## Usage

### Start the Application

```bash
./run_streamlit.sh
# Or manually:
source venv/bin/activate && streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Complete Workflow

#### 1. Create a Job Posting

- Go to **Job Postings** → **New Job**
- Enter job title, city, and minimum repositories
- Paste job description or upload a file (.pdf, .docx, .md, .txt)
- Click "Extract & Preview"
- Edit extracted languages, topics, and requirements using multiselect chips
- Adjust scoring weights (Skills, Activity, Quality, Completeness)
- Click "Save Job Posting"

#### 2. Discover Candidates

- Go to **Candidates**
- Select your job from the dropdown
- Click "Run / Refresh Discovery Now"
- Wait for GitHub search and scoring to complete
- Apply filters:
  - "Has all must-haves" - Only show candidates matching all required skills
  - "Active in last 90 days" - Only show recently active developers

#### 3. Invite Top Candidates

- Use the "Top N" slider to select your preferred number
- Review candidate profiles, scores, and evidence
- Click "Invite" button for selected candidates
- Copy the generated invitation link
- Send the link to candidates (via email, LinkedIn, etc.)

#### 4. Candidate Applies

- Candidate clicks the invitation link
- Fills out profile (name, email, LinkedIn, years of experience)
- Takes Soft Skills assessment (7 minutes)
- Takes Technical assessment (20 minutes)
- System tracks anti-cheat metrics (tab switches, copy/paste attempts)

#### 5. Review Results

- Go to **Tests (HR)**
- Select the job
- View all assessment attempts with:
  - Scores (Soft and Tech)
  - Duration
  - Anti-cheat flags
  - Completion status

#### 6. Track Progress

- **Dashboard** shows real-time metrics:
  - Total job posts
  - Candidates discovered
  - Applications received
  - Tests completed

### Advanced Features

#### Custom Scoring Weights

When creating a job, adjust the weight sliders:
- **Skills** (default 60%): Match on languages and topics
- **Activity** (default 25%): Contributions and recency
- **Quality** (default 10%): Stars and followers
- **Completeness** (default 5%): Profile completeness

#### File Upload Support

Upload job descriptions in multiple formats:
- PDF (`.pdf`)
- Word (`.docx`)
- Markdown (`.md`)
- Plain text (`.txt`)

#### Export Data

From the Candidates page:
- **CSV Export**: Spreadsheet-compatible format
- **JSON Export**: Structured data for integrations

## Architecture

### Single-File Design

The entire application is contained in `streamlit_app.py` (~1800 lines):
- Database models (SQLModel)
- Utility functions
- Gemini AI integration
- GitHub GraphQL queries
- Scoring algorithm
- All 5 pages
- Assessment system

### Database Schema

- **JobPosting**: Job details, parsed requirements, weights
- **Candidate**: GitHub user profile, portfolio
- **JobCandidateMatch**: Per-job scoring and evidence
- **Invitation**: Tokenized invite links
- **AssessmentTemplate**: Test questions (Soft/Tech)
- **AssessmentAttempt**: Test results and anti-cheat data

### Technology Stack

- **Streamlit**: Web framework
- **SQLModel**: Database ORM
- **Google Gemini**: AI text extraction
- **GitHub GraphQL API**: Developer search
- **pypdf**: PDF parsing
- **python-docx**: DOCX parsing

## API Keys

### Google Gemini API

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create an API key
3. Add to `.env` as `GEMINI_API_KEY`

### GitHub Personal Access Token

1. Go to [GitHub Settings → Developer settings → Personal access tokens](https://github.com/settings/tokens)
2. Generate new token (classic)
3. Select scopes: `read:user`, `user:email`, `read:org`
4. Add to `.env` as `GITHUB_TOKEN`

## Limitations

- GitHub rate limits: ~10-30 candidates per discovery run
- No email sending (invitations are copyable links)
- Lightweight anti-cheat (not enterprise-grade proctoring)
- Local SQLite database (single user)

## Troubleshooting

### GitHub Rate Limits

If you see "RESOURCE_LIMITS_EXCEEDED":
- The app already uses minimal queries (10 users/page, 3 pages max)
- Wait a few minutes and try again
- Consider using a GitHub personal access token with higher limits

### File Upload Errors

If PDF or DOCX uploads fail:
```bash
pip install pypdf python-docx
```

### Database Issues

To reset the database:
```bash
rm ats.db
# Restart the app - database will be recreated
```

## License

MIT

## Contributing

This is a minimal ATS demo. For production use, consider:
- PostgreSQL instead of SQLite
- Email integration (SendGrid, AWS SES)
- Enterprise proctoring (e.g., Proctorio API)
- Multi-tenancy support
- Advanced analytics dashboard
