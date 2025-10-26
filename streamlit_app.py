"""
TalentSonar - AI-Powered Applicant Tracking System
Streamlit Web Application - Single File
"""

import os
import json
import asyncio
import secrets
import base64
import warnings
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple
import math
import pandas as pd
from io import StringIO, BytesIO

import streamlit as st
import streamlit.components.v1 as components
import httpx
from dotenv import load_dotenv
import google.generativeai as genai

# SQLModel imports
from sqlmodel import Field, SQLModel, create_engine, Session, select, Column, JSON
from sqlalchemy import func

# Suppress SQLAlchemy warnings during Streamlit hot-reload
warnings.filterwarnings('ignore', category=Warning, module='sqlalchemy')

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="TalentSonar",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure APIs
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


# ============================================================================
# DATA MODELS (SQLModel)
# ============================================================================

class JobPosting(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    city: str
    city_synonyms: str = Field(default="[]")  # JSON string
    min_repos: int = Field(default=5)
    raw_description: str
    parsed_description: str = Field(default="{}")  # JSON string
    weights: str = Field(default='{"skills":60,"activity":25,"quality":10,"completeness":5}')  # JSON
    is_active: bool = Field(default=True)
    num_candidates: int = Field(default=0)
    num_applied: int = Field(default=0)
    num_tested: int = Field(default=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Candidate(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    
    id: Optional[int] = Field(default=None, primary_key=True)
    login: str = Field(unique=True, index=True)
    name: Optional[str] = None
    github_url: Optional[str] = None
    linkedin_url: Optional[str] = None
    email: Optional[str] = None
    location: Optional[str] = None
    followers: int = Field(default=0)
    total_stars: int = Field(default=0)
    years_experience: Optional[int] = None
    portfolio: str = Field(default="[]", sa_column=Column(JSON))  # JSON list of repos
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class JobCandidateMatch(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="jobposting.id", index=True)
    candidate_id: int = Field(foreign_key="candidate.id", index=True)
    langs_found: str = Field(default="[]", sa_column=Column(JSON))
    topics_found: str = Field(default="[]", sa_column=Column(JSON))
    requirement_scores: str = Field(default="{}", sa_column=Column(JSON))
    skill_subscore: float = Field(default=0.0)
    activity_subscore: float = Field(default=0.0)
    quality_subscore: float = Field(default=0.0)
    completeness_subscore: float = Field(default=0.0)
    total_score: float = Field(default=0.0)
    evidence: str = Field(default="[]", sa_column=Column(JSON))  # list of reason strings
    status: str = Field(default="DISCOVERED")  # DISCOVERED, INVITED, APPLIED, etc.
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Invitation(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="jobposting.id")
    candidate_id: int = Field(foreign_key="candidate.id")
    token: str = Field(unique=True, index=True)
    expires_at: datetime
    used_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AssessmentTemplate(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    
    id: Optional[int] = Field(default=None, primary_key=True)
    kind: str  # "SOFT" or "TECH"
    title: str
    questions: str = Field(sa_column=Column(JSON))  # JSON list of question dicts
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AssessmentAttempt(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="jobposting.id")
    candidate_id: int = Field(foreign_key="candidate.id")
    kind: str  # "SOFT" or "TECH"
    answers: str = Field(default="{}", sa_column=Column(JSON))
    soft_score: float = Field(default=0.0)
    tech_score: float = Field(default=0.0)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_sec: int = Field(default=0)
    cheating_flags: str = Field(default="[]", sa_column=Column(JSON))
    max_tab_switches: int = Field(default=0)
    copy_paste_count: int = Field(default=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Database setup
# Use PostgreSQL from environment variable, fallback to SQLite for local dev
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///ats.db")

# PostgreSQL-specific settings
if DATABASE_URL.startswith("postgresql"):
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        pool_pre_ping=True,  # Verify connections before using
        pool_size=10,
        max_overflow=20
    )
else:
    # SQLite settings
    engine = create_engine(DATABASE_URL, echo=False)


def init_db():
    """Initialize database tables."""
    SQLModel.metadata.create_all(engine)


# Initialize DB on import
init_db()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def json_dumps(obj: Any) -> str:
    """Safe JSON serialization."""
    return json.dumps(obj, default=str)


def json_loads(s: str, default: Any = None) -> Any:
    """Safe JSON deserialization."""
    try:
        return json.loads(s) if s else default
    except:
        return default


def make_token(job_id: int, candidate_id: int, ttl_hours: int = 168) -> str:
    """Create a URL-safe token for invitation."""
    data = f"{job_id}:{candidate_id}:{secrets.token_urlsafe(16)}"
    return base64.urlsafe_b64encode(data.encode()).decode()


def parse_token(token: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse invitation token. Returns (job_id, candidate_id) or (None, None)."""
    try:
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        parts = decoded.split(":")
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
    except:
        pass
    return None, None


def recalc_job_stats(job_id: int):
    """Recalculate and update job statistics."""
    with Session(engine) as session:
        job = session.get(JobPosting, job_id)
        if not job:
            return
        
        # Count candidates
        num_candidates = session.exec(
            select(func.count(JobCandidateMatch.id)).where(JobCandidateMatch.job_id == job_id)
        ).one()
        
        # Count applied (status >= APPLIED)
        num_applied = session.exec(
            select(func.count(JobCandidateMatch.id)).where(
                JobCandidateMatch.job_id == job_id,
                JobCandidateMatch.status.in_(["APPLIED", "TESTED", "HIRED"])
            )
        ).one()
        
        # Count tested (distinct candidates with finished attempts)
        num_tested = session.exec(
            select(func.count(func.distinct(AssessmentAttempt.candidate_id))).where(
                AssessmentAttempt.job_id == job_id,
                AssessmentAttempt.finished_at.isnot(None)
            )
        ).one()
        
        job.num_candidates = num_candidates
        job.num_applied = num_applied
        job.num_tested = num_tested
        job.updated_at = datetime.now(timezone.utc)
        
        session.add(job)
        session.commit()


def extract_text_from_upload(uploaded_file) -> str:
    """Extract text from uploaded file (PDF, DOCX, MD, TXT)."""
    filename = uploaded_file.name.lower()
    
    try:
        if filename.endswith('.txt') or filename.endswith('.md'):
            return uploaded_file.read().decode('utf-8')
        
        elif filename.endswith('.pdf'):
            try:
                from pypdf import PdfReader
                pdf = PdfReader(BytesIO(uploaded_file.read()))
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                return text
            except ImportError:
                st.error("pypdf not installed. Install with: pip install pypdf")
                return ""
        
        elif filename.endswith('.docx'):
            try:
                from docx import Document
                doc = Document(BytesIO(uploaded_file.read()))
                text = "\n".join([para.text for para in doc.paragraphs])
                return text
            except ImportError:
                st.error("python-docx not installed. Install with: pip install python-docx")
                return ""
        
        else:
            st.error(f"Unsupported file type: {filename}")
            return ""
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""


# ============================================================================
# GEMINI WRAPPER (reuses existing extract_jd_spec)
# ============================================================================

def gemini_extract_spec(text: str) -> dict:
    """Extract JD spec using Gemini. Returns normalized dict with lowercase tokens."""
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""Analyze this job description and extract structured information in JSON format.
Focus on technical skills, programming languages, frameworks, and technologies.
For languages, use standard names like: Python, JavaScript, Java, Go, TypeScript, etc.
For topics, include frameworks, tools, platforms: React, Django, FastAPI, Docker, Kubernetes, etc.

Job Description:
{text}

Return ONLY a valid JSON object with this exact structure (no markdown, no extra text):
{{
  "role": "job title or role name",
  "languages": ["list", "of", "programming", "languages"],
  "topics": ["list", "of", "frameworks", "tools", "technologies"],
  "must_have": ["list", "of", "required", "skills"],
  "nice_to_have": ["list", "of", "optional", "skills"]
}}"""
    
    response = model.generate_content(prompt)
    response_text = response.text.strip()
    
    # Remove markdown code blocks if present
    if response_text.startswith('```'):
        lines = response_text.split('\n')
        json_lines = []
        in_code_block = False
        for line in lines:
            if line.startswith('```'):
                in_code_block = not in_code_block
                continue
            if in_code_block or not line.startswith('```'):
                json_lines.append(line)
        response_text = '\n'.join(json_lines).strip()
    
    spec = json.loads(response_text)
    
    # Normalize to lowercase
    spec['languages'] = [lang.lower() for lang in spec.get('languages', [])]
    spec['topics'] = [topic.lower() for topic in spec.get('topics', [])]
    spec['must_have'] = [item.lower() for item in spec.get('must_have', [])]
    spec['nice_to_have'] = [item.lower() for item in spec.get('nice_to_have', [])]
    
    return spec


# ============================================================================
# GEMINI: Extract structured spec from JD (ORIGINAL - kept for compatibility)
# ============================================================================

def extract_jd_spec(job_description: str) -> dict:
    """Use Gemini to extract structured requirements from job description."""
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""Analyze this job description and extract structured information in JSON format.
Focus on technical skills, programming languages, frameworks, and technologies.
For languages, use standard names like: Python, JavaScript, Java, Go, TypeScript, etc.
For topics, include frameworks, tools, platforms: React, Django, FastAPI, Docker, Kubernetes, etc.

Job Description:
{job_description}

Return ONLY a valid JSON object with this exact structure (no markdown, no extra text):
{{
  "role": "job title or role name",
  "languages": ["list", "of", "programming", "languages"],
  "topics": ["list", "of", "frameworks", "tools", "technologies"],
  "must_have": ["list", "of", "required", "skills"],
  "nice_to_have": ["list", "of", "optional", "skills"]
}}"""
    
    response = model.generate_content(prompt)
    response_text = response.text.strip()
    
    # Remove markdown code blocks if present
    if response_text.startswith('```'):
        lines = response_text.split('\n')
        json_lines = []
        in_code_block = False
        for line in lines:
            if line.startswith('```'):
                in_code_block = not in_code_block
                continue
            if in_code_block or not line.startswith('```'):
                json_lines.append(line)
        response_text = '\n'.join(json_lines).strip()
    
    spec = json.loads(response_text)
    
    # Normalize languages and topics to lowercase for matching
    spec['languages'] = [lang.lower() for lang in spec.get('languages', [])]
    spec['topics'] = [topic.lower() for topic in spec.get('topics', [])]
    
    return spec


# ============================================================================
# GITHUB: Search and fetch user data
# ============================================================================

async def search_github_users(city: str, cities_synonyms: list, languages: list, min_repos: int) -> list:
    """Search GitHub for users matching criteria."""
    
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    
    all_cities = [city] + cities_synonyms
    all_users = {}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for city_name in all_cities:
            for lang in languages if languages else [""]:
                # Build search query
                query_parts = [f'type:user location:"{city_name}"']
                if lang:
                    query_parts.append(f'language:{lang}')
                query_parts.append(f'repos:>{min_repos}')
                
                search_query = ' '.join(query_parts)
                
                # GitHub GraphQL query - reduced complexity to avoid rate limits
                cursor = None
                has_next = True
                pages_fetched = 0
                max_pages = 3  # Limit pagination to avoid excessive API calls
                
                while has_next and pages_fetched < max_pages:
                    graphql_query = """
                    query($searchQuery: String!, $cursor: String) {
                      search(query: $searchQuery, type: USER, first: 10, after: $cursor) {
                        pageInfo {
                          hasNextPage
                          endCursor
                        }
                        nodes {
                          ... on User {
                            login
                            name
                            location
                            followers {
                              totalCount
                            }
                            repositories(first: 3, orderBy: {field: STARGAZERS, direction: DESC}, isFork: false) {
                              nodes {
                                name
                                primaryLanguage {
                                  name
                                }
                                stargazerCount
                                updatedAt
                                repositoryTopics(first: 5) {
                                  nodes {
                                    topic {
                                      name
                                    }
                                  }
                                }
                              }
                            }
                            contributionsCollection {
                              totalCommitContributions
                              totalPullRequestContributions
                              totalIssueContributions
                            }
                          }
                        }
                      }
                    }
                    """
                    
                    variables = {
                        "searchQuery": search_query,
                        "cursor": cursor
                    }
                    
                    try:
                        response = await client.post(
                            "https://api.github.com/graphql",
                            headers=headers,
                            json={"query": graphql_query, "variables": variables}
                        )
                        
                        if response.status_code != 200:
                            st.error(f"GitHub API error: {response.status_code}")
                            break
                        
                        data = response.json()
                        
                        if "errors" in data:
                            st.warning(f"GraphQL partial errors (continuing): {len(data['errors'])} errors")
                        
                        search_results = data.get("data", {}).get("search", {})
                        if not search_results:
                            break
                            
                        users = search_results.get("nodes", [])
                        
                        # Merge/dedupe users
                        for user in users:
                            if user and user.get("login"):
                                login = user["login"]
                                if login not in all_users:
                                    all_users[login] = user
                        
                        # Check pagination
                        page_info = search_results.get("pageInfo", {})
                        has_next = page_info.get("hasNextPage", False)
                        cursor = page_info.get("endCursor")
                        pages_fetched += 1
                        
                        if not has_next:
                            break
                            
                    except Exception as e:
                        st.error(f"Error fetching users: {e}")
                        has_next = False
    
    return list(all_users.values())


# ============================================================================
# SCORING: Calculate match scores
# ============================================================================

def score_user(user: dict, jd_spec: dict) -> dict:
    """Score a user based on JD requirements (0-100)."""
    
    jd_languages = set(jd_spec.get('languages', []))
    jd_topics = set(jd_spec.get('topics', []))
    
    # Extract user's languages and topics from repos
    user_languages = set()
    user_topics = set()
    max_stars = 0
    most_recent_update = None
    
    repos = user.get('repositories', {}).get('nodes', [])
    for repo in repos:
        if repo:
            # Language
            if repo.get('primaryLanguage') and repo['primaryLanguage'].get('name'):
                user_languages.add(repo['primaryLanguage']['name'].lower())
            
            # Topics
            repo_topics = repo.get('repositoryTopics', {}).get('nodes', [])
            for topic_node in repo_topics:
                if topic_node and topic_node.get('topic'):
                    user_topics.add(topic_node['topic']['name'].lower())
            
            # Stars
            stars = repo.get('stargazerCount', 0)
            max_stars = max(max_stars, stars)
            
            # Updated at
            updated_at = repo.get('updatedAt')
            if updated_at:
                updated_date = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                if most_recent_update is None or updated_date > most_recent_update:
                    most_recent_update = updated_date
    
    # 1. Skill match (60%)
    lang_match = len(jd_languages & user_languages)
    topic_match = len(jd_topics & user_topics)
    total_jd_skills = len(jd_languages) + len(jd_topics)
    
    if total_jd_skills > 0:
        skill_score = ((lang_match + topic_match) / total_jd_skills) * 60
    else:
        skill_score = 0
    
    # 2. Activity/Recency (25%)
    contributions = user.get('contributionsCollection', {})
    commits = contributions.get('totalCommitContributions', 0)
    prs = contributions.get('totalPullRequestContributions', 0)
    issues = contributions.get('totalIssueContributions', 0)
    
    total_activity = commits + prs + issues
    # Cap activity score (log scale)
    activity_base = min(math.log(total_activity + 1) / math.log(1000), 1) * 15
    
    # Recency bonus
    recency_bonus = 0
    if most_recent_update:
        days_ago = (datetime.now(most_recent_update.tzinfo) - most_recent_update).days
        if days_ago <= 90:
            recency_bonus = 10
    
    activity_score = activity_base + recency_bonus
    
    # 3. Quality (10%)
    followers = user.get('followers', {}).get('totalCount', 0)
    # Log-capped quality score
    quality_score = min(math.log(max_stars + followers + 1) / math.log(1000), 1) * 10
    
    # 4. Completeness (5%)
    completeness_score = 0
    if user.get('name'):
        completeness_score += 2.5
    if user.get('location'):
        completeness_score += 2.5
    
    # Total score
    total_score = skill_score + activity_score + quality_score + completeness_score
    total_score = min(max(total_score, 0), 100)  # Clamp to 0-100
    
    # Generate human-readable reasons
    reasons = []
    if lang_match > 0:
        reasons.append(f"{lang_match} language match(es)")
    if topic_match > 0:
        reasons.append(f"{topic_match} topic match(es)")
    if total_activity > 100:
        reasons.append(f"{total_activity} contributions last year")
    if recency_bonus > 0:
        reasons.append("Recent activity (≤90 days)")
    if max_stars > 50:
        reasons.append(f"Top repo has {max_stars} stars")
    if followers > 20:
        reasons.append(f"{followers} followers")
    
    return {
        'rank': 0,  # Will be set later
        'login': user.get('login', ''),
        'name': user.get('name', ''),
        'location': user.get('location', ''),
        'score': round(total_score, 1),
        'reasons': ', '.join(reasons) if reasons else 'No strong signals',
        'languages': sorted(list(user_languages)),
        'topics': sorted(list(user_topics)),
        'followers': followers
    }


# ============================================================================
# GITHUB + SCORING WRAPPER (reuses existing functions)
# ============================================================================

async def run_discovery_for_job(job: JobPosting, parsed: dict, custom_weights: dict = None) -> List[dict]:
    """
    Run GitHub discovery and scoring for a job.
    Returns list of candidate dicts with scoring data.
    Reuses existing search_github_users and score_user functions.
    """
    city_synonyms = json_loads(job.city_synonyms, [])
    languages = parsed.get('languages', [])
    topics = parsed.get('topics', [])
    
    # Search GitHub
    users = await search_github_users(job.city, city_synonyms, languages, job.min_repos)
    
    # Build JD spec for scoring
    jd_spec = {
        'languages': languages,
        'topics': topics,
        'must_have': parsed.get('must_have', []),
        'nice_to_have': parsed.get('nice_to_have', [])
    }
    
    # Score each user
    results = []
    for user in users:
        scored = score_user(user, jd_spec)
        
        # Extract detailed info for DB storage
        user_languages = set()
        user_topics = set()
        portfolio = []
        max_stars = 0
        
        repos = user.get('repositories', {}).get('nodes', [])
        for repo in repos:
            if repo:
                repo_data = {
                    'name': repo.get('name', ''),
                    'stars': repo.get('stargazerCount', 0),
                    'language': None,
                    'topics': [],
                    'updated_at': repo.get('updatedAt', '')
                }
                
                if repo.get('primaryLanguage') and repo['primaryLanguage'].get('name'):
                    lang = repo['primaryLanguage']['name'].lower()
                    user_languages.add(lang)
                    repo_data['language'] = lang
                
                repo_topics = repo.get('repositoryTopics', {}).get('nodes', [])
                for topic_node in repo_topics:
                    if topic_node and topic_node.get('topic'):
                        topic = topic_node['topic']['name'].lower()
                        user_topics.add(topic)
                        repo_data['topics'].append(topic)
                
                max_stars = max(max_stars, repo.get('stargazerCount', 0))
                portfolio.append(repo_data)
        
        # Calculate requirement scores
        requirement_scores = {}
        for req in parsed.get('must_have', []):
            req_lower = req.lower()
            if req_lower in user_languages or req_lower in user_topics:
                requirement_scores[req] = 1.0
            else:
                requirement_scores[req] = 0.0
        
        for req in parsed.get('nice_to_have', []):
            req_lower = req.lower()
            if req_lower in user_languages or req_lower in user_topics:
                requirement_scores[req] = 1.0
            else:
                requirement_scores[req] = 0.0
        
        results.append({
            'login': user.get('login', ''),
            'name': user.get('name', ''),
            'location': user.get('location', ''),
            'followers': user.get('followers', {}).get('totalCount', 0),
            'total_stars': max_stars,
            'portfolio': portfolio,
            'langsFound': sorted(list(user_languages)),
            'topicsFound': sorted(list(user_topics)),
            'requirement_scores': requirement_scores,
            'reasons': scored['reasons'].split(', ') if scored['reasons'] else [],
            'skill_subscore': scored.get('skill_subscore', 0),
            'activity_subscore': scored.get('activity_subscore', 0),
            'quality_subscore': scored.get('quality_subscore', 0),
            'completeness_subscore': scored.get('completeness_subscore', 0),
            'score': scored['score']
        })
    
    return results


# ============================================================================
# ASSESSMENT TEMPLATES
# ============================================================================

def seed_assessment_templates():
    """Seed default assessment templates if they don't exist."""
    with Session(engine) as session:
        # Check if template exists
        soft_exists = session.exec(select(AssessmentTemplate).where(AssessmentTemplate.kind == "SOFT")).first()
        
        if not soft_exists:
            soft_questions = [
                {
                    "id": "s1",
                    "prompt": "How do you prioritize tasks when working on multiple projects?",
                    "type": "mcq",
                    "choices": [
                        "Based on deadlines",
                        "Based on importance to stakeholders",
                        "Using a prioritization framework (e.g., Eisenhower matrix)",
                        "First come, first served"
                    ],
                    "answer": 2
                },
                {
                    "id": "s2",
                    "prompt": "A team member disagrees with your approach. What do you do?",
                    "type": "mcq",
                    "choices": [
                        "Insist on your approach",
                        "Listen to their perspective and discuss pros/cons",
                        "Ask the manager to decide",
                        "Compromise without discussion"
                    ],
                    "answer": 1
                },
                {
                    "id": "s3",
                    "prompt": "How do you handle stress during tight deadlines?",
                    "type": "mcq",
                    "choices": [
                        "Work longer hours",
                        "Break tasks into smaller chunks and focus",
                        "Ask for deadline extension",
                        "Delegate everything"
                    ],
                    "answer": 1
                },
                {
                    "id": "s4",
                    "prompt": "What's your approach to learning new technologies?",
                    "type": "mcq",
                    "choices": [
                        "Read documentation cover-to-cover",
                        "Build a small project immediately",
                        "Take an online course first",
                        "Ask colleagues to teach me"
                    ],
                    "answer": 1
                },
                {
                    "id": "s5",
                    "prompt": "How do you give feedback to peers?",
                    "type": "mcq",
                    "choices": [
                        "Direct and immediate",
                        "Sandwich method (positive-negative-positive)",
                        "Only when asked",
                        "Through the manager"
                    ],
                    "answer": 1
                }
            ]
            
            template = AssessmentTemplate(
                kind="SOFT",
                title="Soft Skills Assessment",
                questions=json_dumps(soft_questions)
            )
            session.add(template)
            session.commit()


# Seed templates on startup
seed_assessment_templates()


# ============================================================================
# STREAMLIT APP - NAVIGATION
# ============================================================================

# ============================================================================
# STREAMLIT APP - NAVIGATION
# ============================================================================

def main():
    """Main application with navigation."""
    
    # Initialize session state
    if 'selected_job_id' not in st.session_state:
        st.session_state.selected_job_id = None
    
    # Check query params for page navigation
    query_params = st.query_params
    query_page = query_params.get("page", None)
    
    # Sidebar navigation
    with st.sidebar:
        # Logo with padding
        st.markdown("<div style='padding: 1rem 0;'>", unsafe_allow_html=True)
        if os.path.exists("TalentSonar.png"):
            st.image("TalentSonar.png", width='stretch')
        else:
            st.title("🎯 TalentSonar")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation header
        st.markdown("### Navigation")
        st.markdown("")
        
        # Navigation options with icons
        nav_items = {
            "Dashboard": "📊",
            "Job Postings": "💼",
            "Candidates": "👥",
            "Assessments": "📝",
            "Candidate Portal": "🔐"
        }
        
        # Set default index based on query param
        page_options = list(nav_items.keys())
        default_index = 0
        if query_page and query_page in page_options:
            default_index = page_options.index(query_page)
        
        # Custom radio with better formatting
        page = st.radio(
            "nav_label",
            page_options,
            index=default_index,
            key="nav_page",
            format_func=lambda x: f"{nav_items[x]} {x}",
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Footer with version and stats
        st.markdown("##### Quick Stats")
        with Session(engine) as session:
            total_jobs = session.exec(select(func.count(JobPosting.id))).first() or 0
            total_candidates = session.exec(select(func.count(Candidate.id))).first() or 0
            total_assessments = session.exec(select(func.count(AssessmentAttempt.id))).first() or 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Jobs", total_jobs)
            st.metric("Candidates", total_candidates)
        with col2:
            st.metric("Assessments", total_assessments)
        
        st.markdown("---")
        st.caption("TalentSonar v1.0")
        st.caption("AI-Powered Recruiting")
    
    # Route to pages (map "Assessments" display name to internal "Tests" page)
    if page == "Dashboard":
        page_dashboard()
    elif page == "Job Postings":
        page_job_postings()
    elif page == "Candidates":
        page_candidates()
    elif page == "Assessments":  # Display name
        page_tests()  # Internal page function
    elif page == "Candidate Portal":
        page_candidate_portal()


# ============================================================================
# PAGE: Dashboard
# ============================================================================

def page_dashboard():
    """Dashboard showing overview metrics and job list."""
    st.title("📊 Dashboard")
    
    with Session(engine) as session:
        jobs = session.exec(select(JobPosting).where(JobPosting.is_active == True)).all()
        
        if not jobs:
            st.info("� Welcome! No active job postings yet.")
            st.markdown("### Get Started")
            st.markdown("1. Go to **Job Postings** to create your first job")
            st.markdown("2. Upload or paste a job description")
            st.markdown("3. Discover candidates from GitHub")
            st.markdown("4. Invite top candidates to apply")
            return
        
        # Metrics
        total_jobs = len(jobs)
        total_candidates = sum(j.num_candidates for j in jobs)
        total_applied = sum(j.num_applied for j in jobs)
        total_tested = sum(j.num_tested for j in jobs)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📋 Job Posts", total_jobs)
        col2.metric("👥 Candidates", total_candidates)
        col3.metric("✅ Applied", total_applied)
        col4.metric("📝 Tested", total_tested)
        
        st.markdown("---")
        
        # Job list table
        st.subheader("Active Job Postings")
        
        job_data = []
        for job in jobs:
            job_data.append({
                "ID": job.id,
                "Title": job.title,
                "City": job.city,
                "Candidates": job.num_candidates,
                "Applied": job.num_applied,
                "Tested": job.num_tested,
                "Created": job.created_at.strftime("%Y-%m-%d")
            })
        
        if job_data:
            df = pd.DataFrame(job_data)
            st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================================
# PAGE: Job Postings
# ============================================================================

def page_job_postings():
    """Job postings management - create and manage jobs."""
    st.title("📋 Job Postings")
    
    tab1, tab2 = st.tabs(["➕ New Job", "📂 Manage"])
    
    # TAB 1: Create new job
    with tab1:
        st.subheader("Create New Job Posting")
        
        with st.form("new_job_form"):
            title = st.text_input("Job Title*", placeholder="e.g., Senior Backend Engineer")
            
            col1, col2 = st.columns(2)
            with col1:
                city = st.text_input("City*", value="Rome")
            with col2:
                city_synonyms_str = st.text_input("City Synonyms (comma-separated)", value="Roma")
            
            min_repos = st.number_input("Minimum Repositories", min_value=1, value=5)
            
            jd_text = st.text_area(
                "Job Description",
                height=200,
                placeholder="Paste job description here..."
            )
            
            uploaded_file = st.file_uploader(
                "Or upload JD file (.pdf, .docx, .md, .txt)",
                type=['pdf', 'docx', 'md', 'txt']
            )
            
            submitted = st.form_submit_button("🔍 Extract & Preview", type="primary")
        
        if submitted:
            if not title.strip():
                st.error("Please enter a job title")
                return
            
            if not city.strip():
                st.error("Please enter a city")
                return
            
            # Get JD text
            final_jd_text = jd_text.strip()
            if uploaded_file and not final_jd_text:
                final_jd_text = extract_text_from_upload(uploaded_file)
            
            if not final_jd_text:
                st.error("Please provide a job description (text or file)")
                return
            
            # Extract with Gemini
            with st.spinner("🤖 Analyzing job description with AI..."):
                try:
                    parsed = gemini_extract_spec(final_jd_text)
                    
                    st.success("✅ Job description extracted!")
                    
                    # Store in session for editing
                    st.session_state['new_job_data'] = {
                        'title': title,
                        'city': city,
                        'city_synonyms': city_synonyms_str,
                        'min_repos': min_repos,
                        'raw_description': final_jd_text,
                        'parsed': parsed
                    }
                    
                except Exception as e:
                    st.error(f"Error extracting JD: {e}")
                    return
        
        # Show editable preview if extracted
        if 'new_job_data' in st.session_state:
            st.markdown("---")
            st.subheader("📝 Edit Extracted Requirements")
            
            data = st.session_state['new_job_data']
            parsed = data['parsed']
            
            st.markdown(f"**Role:** {parsed.get('role', 'N/A')}")
            
            # Editable multiselects
            languages = st.multiselect(
                "Programming Languages",
                options=parsed.get('languages', []) + ['python', 'javascript', 'java', 'go', 'typescript', 'rust', 'c++'],
                default=parsed.get('languages', [])
            )
            
            topics = st.multiselect(
                "Topics/Frameworks",
                options=parsed.get('topics', []) + ['react', 'django', 'fastapi', 'docker', 'kubernetes', 'aws', 'postgresql'],
                default=parsed.get('topics', [])
            )
            
            must_have = st.multiselect(
                "Must Have Skills",
                options=parsed.get('must_have', []) + languages + topics,
                default=parsed.get('must_have', [])
            )
            
            nice_to_have = st.multiselect(
                "Nice to Have Skills",
                options=parsed.get('nice_to_have', []) + languages + topics,
                default=parsed.get('nice_to_have', [])
            )
            
            st.markdown("### ⚖️ Scoring Weights")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                w_skills = st.slider("Skills", 0, 100, 60)
            with col2:
                w_activity = st.slider("Activity", 0, 100, 25)
            with col3:
                w_quality = st.slider("Quality", 0, 100, 10)
            with col4:
                w_completeness = st.slider("Completeness", 0, 100, 5)
            
            total_weight = w_skills + w_activity + w_quality + w_completeness
            if total_weight != 100:
                st.warning(f"⚠️ Weights sum to {total_weight}. Should be 100.")
            
            if st.button("💾 Save Job Posting", type="primary"):
                # Update parsed data
                parsed['languages'] = languages
                parsed['topics'] = topics
                parsed['must_have'] = must_have
                parsed['nice_to_have'] = nice_to_have
                
                weights = {
                    'skills': w_skills,
                    'activity': w_activity,
                    'quality': w_quality,
                    'completeness': w_completeness
                }
                
                city_synonyms = [s.strip() for s in data['city_synonyms'].split(',') if s.strip()]
                
                # Save to DB
                with Session(engine) as session:
                    job = JobPosting(
                        title=data['title'],
                        city=data['city'],
                        city_synonyms=json_dumps(city_synonyms),
                        min_repos=data['min_repos'],
                        raw_description=data['raw_description'],
                        parsed_description=json_dumps(parsed),
                        weights=json_dumps(weights)
                    )
                    session.add(job)
                    session.commit()
                    session.refresh(job)
                    
                    st.success(f"✅ Job posting #{job.id} created!")
                    del st.session_state['new_job_data']
                    st.rerun()
    
    # TAB 2: Manage existing jobs
    with tab2:
        st.subheader("Manage Job Postings")
        
        with Session(engine) as session:
            jobs = session.exec(select(JobPosting).order_by(JobPosting.created_at.desc())).all()
            
            if not jobs:
                st.info("No job postings yet. Create one in the 'New Job' tab!")
                return
            
            # Display jobs as cards
            for job in jobs:
                with st.expander(f"**{job.title}** (ID: {job.id}) - {job.city}", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Candidates", job.num_candidates)
                    col2.metric("Applied", job.num_applied)
                    col3.metric("Tested", job.num_tested)
                    col4.metric("Min Repos", job.min_repos)
                    
                    st.markdown(f"**Created:** {job.created_at.strftime('%Y-%m-%d %H:%M')}")
                    st.markdown(f"**Active:** {'✅ Yes' if job.is_active else '❌ No'}")
                    
                    if st.button(f"📊 Open Dataset", key=f"open_{job.id}"):
                        st.session_state.selected_job_id = job.id
                        st.query_params["page"] = "Candidates"
                        st.rerun()
                    
                    if st.button(f"🗑️ Delete", key=f"del_{job.id}"):
                        session.delete(job)
                        session.commit()
                        st.success(f"Deleted job #{job.id}")
                        st.rerun()


# ============================================================================
# PAGE: Candidates
# ============================================================================

def page_candidates():
    """Candidates page - discovery and management per job."""
    st.title("👥 Candidates")
    
    with Session(engine) as session:
        jobs = session.exec(select(JobPosting).where(JobPosting.is_active == True)).all()
        
        if not jobs:
            st.info("No active jobs. Create one in Job Postings first!")
            return
        
        # Job selector
        job_options = {f"{j.id}: {j.title}": j.id for j in jobs}
        
        # Use selected_job_id from session if available
        default_idx = 0
        if st.session_state.selected_job_id:
            for idx, (label, jid) in enumerate(job_options.items()):
                if jid == st.session_state.selected_job_id:
                    default_idx = idx
                    break
        
        selected_label = st.selectbox(
            "Select Job",
            options=list(job_options.keys()),
            index=default_idx
        )
        
        job_id = job_options[selected_label]
        st.session_state.selected_job_id = job_id
        
        job = session.get(JobPosting, job_id)
        if not job:
            st.error("Job not found")
            return
        
        parsed = json_loads(job.parsed_description, {})
        city_synonyms = json_loads(job.city_synonyms, [])
        
        # Show config
        with st.expander("⚙️ Job Configuration", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("City", job.city)
            col2.metric("City Synonyms", len(city_synonyms))
            col3.metric("Min Repos", job.min_repos)
            
            st.markdown(f"**Languages:** {', '.join(parsed.get('languages', []))}")
            st.markdown(f"**Topics:** {', '.join(parsed.get('topics', []))}")
            st.markdown(f"**Must Have:** {', '.join(parsed.get('must_have', []))}")
        
        # Discovery button
        if st.button("🔍 Run / Refresh Discovery Now", type="primary"):
            with st.spinner("Searching GitHub and scoring candidates..."):
                try:
                    # Run discovery
                    results = asyncio.run(run_discovery_for_job(job, parsed))
                    
                    # Upsert candidates and matches
                    for res in results:
                        # Upsert candidate
                        cand = session.exec(
                            select(Candidate).where(Candidate.login == res['login'])
                        ).first()
                        
                        if not cand:
                            cand = Candidate(
                                login=res['login'],
                                name=res['name'],
                                github_url=f"https://github.com/{res['login']}",
                                location=res['location'],
                                followers=res['followers'],
                                total_stars=res['total_stars'],
                                portfolio=json_dumps(res['portfolio'])
                            )
                            session.add(cand)
                            session.commit()
                            session.refresh(cand)
                        else:
                            # Update existing
                            cand.name = res['name'] or cand.name
                            cand.location = res['location'] or cand.location
                            cand.followers = res['followers']
                            cand.total_stars = res['total_stars']
                            cand.portfolio = json_dumps(res['portfolio'])
                            cand.updated_at = datetime.now(timezone.utc)
                            session.add(cand)
                            session.commit()
                        
                        # Upsert match
                        match = session.exec(
                            select(JobCandidateMatch).where(
                                JobCandidateMatch.job_id == job_id,
                                JobCandidateMatch.candidate_id == cand.id
                            )
                        ).first()
                        
                        if not match:
                            match = JobCandidateMatch(
                                job_id=job_id,
                                candidate_id=cand.id,
                                langs_found=json_dumps(res['langsFound']),
                                topics_found=json_dumps(res['topicsFound']),
                                requirement_scores=json_dumps(res['requirement_scores']),
                                skill_subscore=res.get('skill_subscore', 0),
                                activity_subscore=res.get('activity_subscore', 0),
                                quality_subscore=res.get('quality_subscore', 0),
                                completeness_subscore=res.get('completeness_subscore', 0),
                                total_score=res['score'],
                                evidence=json_dumps(res['reasons'])
                            )
                            session.add(match)
                        else:
                            match.langs_found = json_dumps(res['langsFound'])
                            match.topics_found = json_dumps(res['topicsFound'])
                            match.requirement_scores = json_dumps(res['requirement_scores'])
                            match.skill_subscore = res.get('skill_subscore', 0)
                            match.activity_subscore = res.get('activity_subscore', 0)
                            match.quality_subscore = res.get('quality_subscore', 0)
                            match.completeness_subscore = res.get('completeness_subscore', 0)
                            match.total_score = res['score']
                            match.evidence = json_dumps(res['reasons'])
                            match.updated_at = datetime.now(timezone.utc)
                            session.add(match)
                        
                        session.commit()
                    
                    recalc_job_stats(job_id)
                    st.success(f"✅ Discovery complete! Found {len(results)} candidates.")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error during discovery: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Get matches
        matches = session.exec(
            select(JobCandidateMatch, Candidate).where(
                JobCandidateMatch.job_id == job_id
            ).join(Candidate).order_by(JobCandidateMatch.total_score.desc())
        ).all()
        
        if not matches:
            st.info("No candidates yet. Run discovery above!")
            return
        
        st.markdown("---")
        st.subheader(f"📊 Dataset ({len(matches)} candidates)")
        
        # Filters
        st.markdown("### 🔧 Filters")
        col1, col2 = st.columns(2)
        
        with col1:
            filter_must_have = st.checkbox("Has all must-haves", value=False)
        
        with col2:
            filter_active_90d = st.checkbox("Active in last 90 days", value=False)
        
        # Build table data
        table_data = []
        for match, cand in matches:
            langs_found = json_loads(match.langs_found, [])
            topics_found = json_loads(match.topics_found, [])
            portfolio = json_loads(cand.portfolio, [])
            
            # Filter: must-haves
            if filter_must_have:
                must_haves = parsed.get('must_have', [])
                if must_haves:
                    all_skills = set(langs_found + topics_found)
                    if not all(mh.lower() in all_skills for mh in must_haves):
                        continue
            
            # Filter: active 90d
            if filter_active_90d:
                is_active = False
                for repo in portfolio:
                    updated_str = repo.get('updated_at', '')
                    if updated_str:
                        try:
                            updated_date = datetime.fromisoformat(updated_str.replace('Z', '+00:00'))
                            days_ago = (datetime.now(timezone.utc) - updated_date).days
                            if days_ago <= 90:
                                is_active = True
                                break
                        except:
                            pass
                if not is_active:
                    continue
            
            table_data.append({
                'match': match,
                'cand': cand,
                'login': cand.login,
                'name': cand.name or '',
                'location': cand.location or '',
                'followers': cand.followers,
                'stars': cand.total_stars,
                'score': match.total_score,
                'langs': ', '.join(langs_found),
                'topics': ', '.join(topics_found),
                'email': cand.email or '',
                'linkedin': cand.linkedin_url or ''
            })
        
        st.markdown(f"Showing {len(table_data)} candidates")
        
        # Top-N selector
        st.markdown("### 🏆 Top N Candidates")
        top_n = st.slider("Select Top N", 1, min(50, len(table_data)), min(10, len(table_data)))
        
        top_candidates = table_data[:top_n]
        
        for idx, item in enumerate(top_candidates, 1):
            match = item['match']
            cand = item['cand']
            evidence = json_loads(match.evidence, [])
            
            with st.container():
                col1, col2, col3 = st.columns([1, 4, 2])
                
                with col1:
                    st.markdown(f"### #{idx}")
                    score_color = "🟢" if match.total_score >= 70 else "🟡" if match.total_score >= 40 else "⚪"
                    st.markdown(f"{score_color} **{match.total_score:.1f}**")
                
                with col2:
                    st.markdown(f"### [{cand.login}](https://github.com/{cand.login})")
                    if cand.name:
                        st.markdown(f"*{cand.name}*")
                    st.markdown(f"📍 {cand.location or 'N/A'}")
                    if evidence:
                        st.markdown(f"**Evidence:** {', '.join(evidence)}")
                    st.markdown(f"**Languages:** {item['langs']}")
                    st.markdown(f"**Topics:** {item['topics']}")
                
                with col3:
                    st.metric("Followers", cand.followers)
                    st.metric("Stars", cand.total_stars)
                    
                    # Invite button
                    if st.button(f"✉️ Invite", key=f"invite_{match.id}"):
                        # Create invitation
                        token = make_token(job.id, cand.id)
                        expires = datetime.now(timezone.utc) + timedelta(days=7)
                        
                        invite = Invitation(
                            job_id=job.id,
                            candidate_id=cand.id,
                            token=token,
                            expires_at=expires
                        )
                        session.add(invite)
                        
                        # Update match status
                        match.status = "INVITED"
                        match.updated_at = datetime.now(timezone.utc)
                        session.add(match)
                        
                        session.commit()
                        
                        # Show copyable link
                        invite_url = f"http://localhost:8501/?page=Candidate%20Portal&token={token}"
                        st.success("✅ Invitation created!")
                        st.code(invite_url, language="text")
                        st.caption("Copy this link and send it to the candidate")
                
                st.markdown("---")
        
        # Full dataset table
        st.markdown("### 📋 Full Dataset Table")
        if table_data:
            df = pd.DataFrame([{
                'Login': item['login'],
                'Name': item['name'],
                'Location': item['location'],
                'Score': f"{item['score']:.1f}",
                'Followers': item['followers'],
                'Stars': item['stars'],
                'Languages': item['langs'],
                'Topics': item['topics']
            } for item in table_data])
            
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Export buttons
        st.markdown("### 📥 Export")
        col1, col2 = st.columns(2)
        
        with col1:
            if table_data:
                csv_data = pd.DataFrame([{
                    'Login': item['login'],
                    'Name': item['name'],
                    'Location': item['location'],
                    'Score': item['score'],
                    'Followers': item['followers'],
                    'Stars': item['stars'],
                    'Languages': item['langs'],
                    'Topics': item['topics'],
                    'Email': item['email'],
                    'LinkedIn': item['linkedin']
                } for item in table_data]).to_csv(index=False)
                
                st.download_button(
                    "📥 Download CSV",
                    data=csv_data,
                    file_name=f"candidates_job_{job_id}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if table_data:
                json_data = json_dumps([{
                    'login': item['login'],
                    'name': item['name'],
                    'location': item['location'],
                    'score': item['score'],
                    'followers': item['followers'],
                    'stars': item['stars'],
                    'languages': item['langs'],
                    'topics': item['topics'],
                    'email': item['email'],
                    'linkedin': item['linkedin']
                } for item in table_data])
                
                st.download_button(
                    "📥 Download JSON",
                    data=json_data,
                    file_name=f"candidates_job_{job_id}.json",
                    mime="application/json"
                )


# ============================================================================
# PAGE: Tests (HR View)
# ============================================================================

def page_tests():
    """Tests page - HR view of assessment attempts."""
    st.title("📝 Assessments (HR View)")
    
    with Session(engine) as session:
        jobs = session.exec(select(JobPosting).where(JobPosting.is_active == True)).all()
        
        if not jobs:
            st.info("No active jobs yet!")
            return
        
        # Job selector
        job_options = {f"{j.id}: {j.title}": j.id for j in jobs}
        selected_label = st.selectbox("Select Job", options=list(job_options.keys()))
        job_id = job_options[selected_label]
        
        # Get attempts for this job
        attempts = session.exec(
            select(AssessmentAttempt, Candidate).where(
                AssessmentAttempt.job_id == job_id
            ).join(Candidate).order_by(AssessmentAttempt.created_at.desc())
        ).all()
        
        if not attempts:
            st.info("No test attempts for this job yet.")
            return
        
        st.subheader(f"📊 Assessment Attempts ({len(attempts)})")
        
        # Build table
        table_data = []
        for attempt, cand in attempts:
            table_data.append({
                'Login': cand.login,
                'Name': cand.name or '',
                'Score': f"{attempt.soft_score:.1f}/10",
                'Duration (sec)': attempt.duration_sec,
                'Tab Switches': attempt.max_tab_switches,
                'Copy/Paste': attempt.copy_paste_count,
                'Finished': "✅" if attempt.finished_at else "⏳",
                'Started': attempt.started_at.strftime("%Y-%m-%d %H:%M") if attempt.started_at else ""
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================================
# PAGE: Candidate Portal
# ============================================================================

def page_candidate_portal():
    """Candidate portal - accessed via invite token."""
    st.title("🎯 Candidate Portal")
    
    # Check for token in query params
    query_params = st.query_params
    token = query_params.get("token", None)
    
    if not token:
        st.warning("⚠️ No invitation token provided.")
        st.info("This page is accessed via an invitation link sent by the recruiter.")
        return
    
    # Parse token
    job_id, candidate_id = parse_token(token)
    
    if not job_id or not candidate_id:
        st.error("❌ Invalid invitation token.")
        return
    
    with Session(engine) as session:
        # Validate invitation
        invite = session.exec(
            select(Invitation).where(Invitation.token == token)
        ).first()
        
        if not invite:
            st.error("❌ Invitation not found.")
            return
        
        # Check expiration (handle both naive and aware datetimes)
        now = datetime.now(timezone.utc)
        expires_at = invite.expires_at
        
        # If expires_at is naive, make it aware (assume UTC)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        
        if expires_at < now:
            st.error("❌ This invitation has expired.")
            return
        
        # Load job and candidate
        job = session.get(JobPosting, job_id)
        cand = session.get(Candidate, candidate_id)
        
        if not job or not cand:
            st.error("❌ Job or candidate not found.")
            return
        
        parsed = json_loads(job.parsed_description, {})
        
        st.success(f"✅ Welcome! Invitation for: **{job.title}** in **{job.city}**")
        
        st.markdown("---")
        
        # Profile section
        st.subheader("👤 Your Profile")
        
        with st.form("profile_form"):
            name = st.text_input("Full Name", value=cand.name or "")
            email = st.text_input("Email", value=cand.email or "")
            linkedin = st.text_input("LinkedIn URL", value=cand.linkedin_url or "")
            years_exp = st.number_input("Years of Experience", min_value=0, max_value=50, value=cand.years_experience or 0)
            
            if st.form_submit_button("💾 Save Profile"):
                cand.name = name
                cand.email = email
                cand.linkedin_url = linkedin
                cand.years_experience = years_exp
                cand.updated_at = datetime.now(timezone.utc)
                session.add(cand)
                session.commit()
                
                # Mark invitation as used if first time
                if not invite.used_at:
                    invite.used_at = datetime.now(timezone.utc)
                    session.add(invite)
                    session.commit()
                
                # Update match status to APPLIED
                match = session.exec(
                    select(JobCandidateMatch).where(
                        JobCandidateMatch.job_id == job_id,
                        JobCandidateMatch.candidate_id == candidate_id
                    )
                ).first()
                
                if match and match.status == "INVITED":
                    match.status = "APPLIED"
                    match.updated_at = datetime.now(timezone.utc)
                    session.add(match)
                    session.commit()
                
                st.success("✅ Profile saved!")
                st.rerun()
        
        st.markdown("---")
        
        # Assessments
        st.subheader("📝 Assessment")
        
        st.info("Complete the soft skills assessment to finish your application.")
        
        st.markdown("### 🤝 Soft Skills Test")
        st.markdown("**Duration:** 7 minutes")
        st.markdown("**Questions:** 5 multiple choice")
        
        # Check if already taken
        soft_attempt = session.exec(
            select(AssessmentAttempt).where(
                AssessmentAttempt.job_id == job_id,
                AssessmentAttempt.candidate_id == candidate_id,
                AssessmentAttempt.kind == "SOFT",
                AssessmentAttempt.finished_at.isnot(None)
            )
        ).first()
        
        if soft_attempt:
            st.success(f"✅ Completed - Score: {soft_attempt.soft_score:.1f}/10")
            st.markdown(f"**Duration:** {soft_attempt.duration_sec} seconds")
            if soft_attempt.max_tab_switches > 0 or soft_attempt.copy_paste_count > 0:
                st.warning(f"⚠️ Tab switches: {soft_attempt.max_tab_switches}, Copy/Paste attempts: {soft_attempt.copy_paste_count}")
        else:
            if st.button("▶️ Start Soft Skills Test", type="primary"):
                st.session_state['active_test'] = 'SOFT'
                st.session_state['test_start_time'] = datetime.now(timezone.utc)
                st.session_state['tab_switches'] = 0
                st.session_state['copy_paste_count'] = 0
                st.rerun()
        
        # Run active test
        if 'active_test' in st.session_state:
            run_assessment(job_id, candidate_id, st.session_state['active_test'], session)


def run_assessment(job_id: int, candidate_id: int, kind: str, session):
    """Run an assessment test with timer and anti-cheat."""
    st.markdown("---")
    st.subheader(f"📝 Soft Skills Assessment")
    
    # Get template
    template = session.exec(
        select(AssessmentTemplate).where(AssessmentTemplate.kind == kind)
    ).first()
    
    if not template:
        st.error("Assessment template not found")
        return
    
    questions = json_loads(template.questions, [])
    duration_minutes = 7  # Only soft skills, always 7 minutes
    
    # Timer
    start_time = st.session_state.get('test_start_time')
    if not start_time:
        st.session_state['test_start_time'] = datetime.now(timezone.utc)
        start_time = st.session_state['test_start_time']
    
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    remaining = (duration_minutes * 60) - elapsed
    
    if remaining <= 0:
        st.error("⏰ Time's up! Auto-submitting...")
        # Auto-submit with current answers
        submit_assessment(job_id, candidate_id, kind, st.session_state.get('test_answers', {}), session)
        return
    
    # Display timer
    mins = int(remaining // 60)
    secs = int(remaining % 60)
    st.warning(f"⏱️ Time Remaining: {mins:02d}:{secs:02d}")
    
    # Anti-cheat display
    col1, col2 = st.columns(2)
    col1.metric("Tab Switches", st.session_state.get('tab_switches', 0))
    col2.metric("Copy/Paste Attempts", st.session_state.get('copy_paste_count', 0))
    
    # Lightweight anti-cheat JS
    anti_cheat_js = """
    <script>
    let tabSwitches = 0;
    let copyPasteCount = 0;
    
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
            tabSwitches++;
            console.log('Tab switch detected:', tabSwitches);
        }
    });
    
    document.addEventListener('copy', function(e) {
        e.preventDefault();
        copyPasteCount++;
        console.log('Copy blocked:', copyPasteCount);
    });
    
    document.addEventListener('paste', function(e) {
        e.preventDefault();
        copyPasteCount++;
        console.log('Paste blocked:', copyPasteCount);
    });
    </script>
    """
    components.html(anti_cheat_js, height=0)
    
    # Questions
    st.markdown("### Questions")
    
    if 'test_answers' not in st.session_state:
        st.session_state['test_answers'] = {}
    
    for i, q in enumerate(questions):
        st.markdown(f"**Q{i+1}.** {q['prompt']}")
        
        answer = st.radio(
            f"Select your answer for Q{i+1}",
            options=q['choices'],
            key=f"q_{q['id']}",
            label_visibility="collapsed"
        )
        
        st.session_state['test_answers'][q['id']] = q['choices'].index(answer)
        st.markdown("---")
    
    # Submit button
    if st.button("✅ Submit Assessment", type="primary"):
        submit_assessment(job_id, candidate_id, kind, st.session_state['test_answers'], session)


def submit_assessment(job_id: int, candidate_id: int, kind: str, answers: dict, session):
    """Submit and score assessment."""
    
    # Get template
    template = session.exec(
        select(AssessmentTemplate).where(AssessmentTemplate.kind == kind)
    ).first()
    
    if not template:
        st.error("Template not found")
        return
    
    questions = json_loads(template.questions, [])
    
    # Score answers (only soft skills)
    correct = 0
    for q in questions:
        user_answer = answers.get(q['id'])
        if user_answer == q['answer']:
            correct += 1
    
    # Calculate score (5 questions, 2 points each, max 10)
    soft_score = min(correct * 2, 10)
    
    # Apply anti-cheat penalties
    tab_switches = st.session_state.get('tab_switches', 0)
    copy_paste = st.session_state.get('copy_paste_count', 0)
    
    penalty = 0
    cheating_flags = []
    
    if tab_switches > 2:
        penalty += 1
        cheating_flags.append(f"Excessive tab switches: {tab_switches}")
    
    if copy_paste > 0:
        penalty += 1
        cheating_flags.append(f"Copy/paste attempts: {copy_paste}")
    
    soft_score = max(0, soft_score - penalty)
    
    # Calculate duration
    start_time = st.session_state.get('test_start_time')
    duration_sec = int((datetime.now(timezone.utc) - start_time).total_seconds())
    
    # Save attempt
    attempt = AssessmentAttempt(
        job_id=job_id,
        candidate_id=candidate_id,
        kind=kind,
        answers=json_dumps(answers),
        soft_score=soft_score,
        tech_score=0,  # Not used anymore
        started_at=start_time,
        finished_at=datetime.now(timezone.utc),
        duration_sec=duration_sec,
        cheating_flags=json_dumps(cheating_flags),
        max_tab_switches=tab_switches,
        copy_paste_count=copy_paste
    )
    
    session.add(attempt)
    session.commit()
    
    # Update match status
    match = session.exec(
        select(JobCandidateMatch).where(
            JobCandidateMatch.job_id == job_id,
            JobCandidateMatch.candidate_id == candidate_id
        )
    ).first()
    
    if match:
        match.status = "TESTED"
        match.updated_at = datetime.now(timezone.utc)
        session.add(match)
        session.commit()
    
    # Recalc job stats
    recalc_job_stats(job_id)
    
    # Clear test state
    for key in ['active_test', 'test_start_time', 'test_answers', 'tab_switches', 'copy_paste_count']:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success(f"✅ Assessment submitted! Score: {soft_score:.1f}/10")
    st.balloons()
    st.rerun()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
