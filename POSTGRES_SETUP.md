# PostgreSQL Setup for TalentSonar

## Local Development Setup

### 1. Install PostgreSQL

**macOS (using Homebrew):**
```bash
brew install postgresql@15
brew services start postgresql@15
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

**Windows:**
Download and install from [postgresql.org](https://www.postgresql.org/download/windows/)

### 2. Create Database

```bash
# Connect to PostgreSQL
psql postgres

# Create database and user
CREATE DATABASE talentsonar;
CREATE USER talentsonar_user WITH ENCRYPTED PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE talentsonar TO talentsonar_user;

# Grant schema privileges (PostgreSQL 15+)
\c talentsonar
GRANT ALL ON SCHEMA public TO talentsonar_user;

# Exit
\q
```

### 3. Update .env File

Add to your `.env` file:

```bash
# For local development
DATABASE_URL=postgresql://talentsonar_user:your_secure_password@localhost:5432/talentsonar

# Or use default user (simpler for local dev)
DATABASE_URL=postgresql://localhost:5432/talentsonar
```

### 4. Install Python Dependencies

```bash
pip install psycopg2-binary
```

### 5. Run Application

```bash
streamlit run streamlit_app.py
```

The app will automatically create all tables on first run!

---

## Production Deployment

### Hosted PostgreSQL Options:

1. **Neon (Recommended - Free Tier)**
   - Sign up at [neon.tech](https://neon.tech)
   - Create a new project
   - Copy the connection string
   - Add to your `.env` or deployment platform

2. **Supabase (Free Tier)**
   - Sign up at [supabase.com](https://supabase.com)
   - Create project → Get connection string
   - Use format: `postgresql://postgres:[password]@[host]:5432/postgres`

3. **Railway.app (Free Tier)**
   - Sign up at [railway.app](https://railway.app)
   - Add PostgreSQL plugin
   - Copy DATABASE_URL from variables

4. **Heroku Postgres**
   - Add Heroku Postgres addon
   - DATABASE_URL automatically set

5. **AWS RDS / Google Cloud SQL / Azure Database**
   - For production scale

### Environment Variable Setup:

**Streamlit Cloud:**
1. Go to App Settings → Secrets
2. Add:
```toml
DATABASE_URL = "postgresql://user:password@host:5432/dbname"
GEMINI_API_KEY = "your_key"
GITHUB_TOKEN = "your_token"
```

**Heroku:**
```bash
heroku config:set DATABASE_URL="postgresql://..."
```

**Docker:**
```yaml
environment:
  - DATABASE_URL=postgresql://user:password@postgres:5432/talentsonar
```

---

## Migration from SQLite

If you have existing SQLite data and want to migrate:

### Option 1: Manual Export/Import

```bash
# Export SQLite data to CSV
sqlite3 ats.db << EOF
.headers on
.mode csv
.output jobposting.csv
SELECT * FROM jobposting;
.output candidate.csv
SELECT * FROM candidate;
-- Repeat for other tables
EOF

# Import to PostgreSQL (create import script)
```

### Option 2: Start Fresh

Simply delete `ats.db` and let the app create new PostgreSQL tables.

---

## Troubleshooting

### Connection Errors

**Error: `could not connect to server`**
```bash
# Check if PostgreSQL is running
brew services list  # macOS
sudo systemctl status postgresql  # Linux

# Start if not running
brew services start postgresql@15
```

**Error: `FATAL: password authentication failed`**
- Check username and password in DATABASE_URL
- Verify user exists: `psql -U postgres -c "\du"`

**Error: `FATAL: database "talentsonar" does not exist`**
```bash
createdb talentsonar
```

### Permission Errors (PostgreSQL 15+)

```sql
-- Connect as superuser
psql -U postgres

-- Grant schema permissions
\c talentsonar
GRANT ALL ON SCHEMA public TO talentsonar_user;
GRANT ALL ON ALL TABLES IN SCHEMA public TO talentsonar_user;
```

### Pool Connection Issues

If you see "too many connections":
- Reduce `pool_size` in streamlit_app.py
- Check max_connections: `psql -c "SHOW max_connections;"`

---

## Benefits of PostgreSQL

✅ **Production-ready**: Handle thousands of concurrent users
✅ **Better performance**: Especially for complex queries
✅ **Data integrity**: ACID compliance, foreign keys enforced
✅ **Scalability**: Can grow from 1 to 1,000,000+ records
✅ **Cloud-native**: Easy to deploy on any platform
✅ **Advanced features**: Full-text search, JSON support, etc.

---

## Reverting to SQLite

If you want to go back to SQLite, simply remove or comment out DATABASE_URL from .env:

```bash
# .env file
# DATABASE_URL=postgresql://...  # Commented out
```

The app will automatically fall back to SQLite!
