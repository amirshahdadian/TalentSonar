# Quick Start Guide - JD2GH Mini-ATS

## Installation (5 minutes)

```bash
# 1. Navigate to project
cd /Users/amirsh/Documents/Repo/hack

# 2. Activate virtual environment (already created)
source venv/bin/activate

# 3. Verify installation
pip list | grep -E "streamlit|sqlmodel|pypdf|python-docx"

# If any missing, install:
pip install -r requirements.txt

# 4. Check environment variables
cat .env
# Should contain:
# GEMINI_API_KEY=your_key_here
# GITHUB_TOKEN=your_token_here
```

## First Run (2 minutes)

```bash
# Start the app
./run_streamlit.sh

# Or manually:
streamlit run streamlit_app.py

# Opens automatically at http://localhost:8501
```

## Quick Test (10 minutes)

### 1. Create a Job (3 min)

1. Click **Job Postings** in sidebar
2. Go to **New Job** tab
3. Fill in:
   - **Title:** "Senior Python Developer"
   - **City:** "Rome"
   - **City Synonyms:** "Roma"
   - **Min Repos:** 5
4. Paste this sample JD:
   ```
   We're looking for a Senior Python Developer with FastAPI experience.
   
   Required:
   - 5+ years Python
   - FastAPI or Django
   - PostgreSQL
   - Docker
   - Git
   
   Nice to have:
   - React
   - AWS
   - GraphQL
   ```
5. Click **🔍 Extract & Preview**
6. Review extracted languages/topics
7. Adjust weight sliders if needed (default: 60/25/10/5)
8. Click **💾 Save Job Posting**

### 2. Find Candidates (3 min)

1. Click **Candidates** in sidebar
2. Select your job from dropdown
3. Click **🔍 Run / Refresh Discovery Now**
4. Wait ~30 seconds for GitHub search
5. See results table with scores
6. Try filters:
   - Check "Has all must-haves"
   - Adjust "Top N" slider to 5

### 3. Invite a Candidate (1 min)

1. Scroll to top candidate
2. Click **✉️ Invite** button
3. Copy the generated link:
   ```
   http://localhost:8501/?page=Candidate%20Portal&token=...
   ```
4. Open link in new browser tab (simulates candidate)

### 4. Candidate Flow (3 min)

1. In the new tab (Candidate Portal):
2. Fill profile form:
   - Name: "Test Candidate"
   - Email: "test@example.com"
   - LinkedIn: "linkedin.com/in/test"
   - Years: 5
3. Click **💾 Save Profile**
4. Click **▶️ Start Soft Skills Test**
5. Answer 5 questions (any answers)
6. Click **✅ Submit Assessment**
7. Click **▶️ Start Technical Test**
8. Answer 8 questions
9. Submit

### 5. Review Results (2 min)

1. Go back to original tab
2. Click **Tests** in sidebar
3. Select your job
4. See test results table with:
   - Scores (Soft/Tech)
   - Duration
   - Anti-cheat counters

### 6. Check Dashboard (1 min)

1. Click **Dashboard** in sidebar
2. See metrics:
   - 1 Job Post
   - X Candidates found
   - 1 Applied
   - 1 Tested

## Common Commands

```bash
# Start app
./run_streamlit.sh

# Stop app
# Press Ctrl+C in terminal

# Reset database
rm ats.db
# Then restart app

# View database
sqlite3 ats.db
sqlite> .tables
sqlite> SELECT * FROM jobposting;
sqlite> .exit

# Check logs
# Streamlit shows logs in terminal

# Update dependencies
pip install -r requirements.txt --upgrade
```

## Troubleshooting

### App won't start
```bash
# Check Python version (need 3.9+)
python --version

# Reinstall dependencies
pip install -r requirements.txt

# Check for port conflicts
lsof -i :8501
# Kill if needed: kill -9 <PID>
```

### API Errors

**Gemini API Error:**
```bash
# Check .env file
cat .env | grep GEMINI
# Should show: GEMINI_API_KEY=...

# Test key
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Key:', os.getenv('GEMINI_API_KEY')[:10]+'...')"
```

**GitHub Rate Limit:**
- Wait 60 minutes
- Or use a different GitHub token
- Discovery is limited to ~30 candidates per run (by design)

### Database Issues

```bash
# Check if database exists
ls -lh ats.db

# Reset completely
rm ats.db
streamlit run streamlit_app.py
# Database recreated automatically
```

### File Upload Not Working

```bash
# Install file parsing libraries
pip install pypdf python-docx

# Verify installation
python -c "import pypdf; import docx; print('OK')"
```

## Tips

1. **First Job:** Use the sample JD provided in the form
2. **Test Portal:** Open invite links in incognito/private window
3. **Anti-Cheat:** Switch tabs during test to see counters increment
4. **Export:** Download CSV from Candidates page for Excel analysis
5. **Multiple Jobs:** Create 2-3 jobs to see Dashboard metrics
6. **Filters:** Use "Has all must-haves" to find perfect matches

## Next Steps

1. **Read Full Docs:** See README.md for complete features
2. **Implementation Details:** See IMPLEMENTATION.md for architecture
3. **Customize Tests:** Edit `seed_assessment_templates()` in streamlit_app.py
4. **Add Questions:** Modify the questions list in the function
5. **Change Weights:** Adjust default weights in job creation form

## File Locations

```
streamlit_app.py    # Main app (edit this)
ats.db              # Database (auto-created)
.env                # API keys (NEVER commit)
requirements.txt    # Dependencies
README.md           # Full documentation
IMPLEMENTATION.md   # Technical details
```

## Support

- Check errors in terminal where Streamlit is running
- Use browser console (F12) to see JavaScript anti-cheat logs
- Database schema: See IMPLEMENTATION.md
- API docs: See README.md

## Production Notes

This is a **development/demo** setup. For production:

1. Change database to PostgreSQL
2. Add authentication (Streamlit auth or OAuth)
3. Use environment-specific configs
4. Add email service (SendGrid, AWS SES)
5. Deploy to Streamlit Cloud, Heroku, or AWS
6. Add logging and monitoring
7. Implement proper security (HTTPS, CSP headers)
8. Rate limiting on API calls
9. Data backup strategy
10. GDPR compliance (data retention, deletion)
