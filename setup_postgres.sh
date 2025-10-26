#!/bin/bash
# Quick PostgreSQL setup script for TalentSonar

echo "🎯 TalentSonar PostgreSQL Setup"
echo "================================"
echo ""

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL is not installed."
    echo ""
    echo "Install with:"
    echo "  macOS:   brew install postgresql@15"
    echo "  Ubuntu:  sudo apt install postgresql postgresql-contrib"
    echo ""
    exit 1
fi

echo "✅ PostgreSQL found"
echo ""

# Database configuration
DB_NAME="talentsonar"
DB_USER="talentsonar_user"
DB_PASSWORD="talentsonar_dev_password"

echo "Creating database: $DB_NAME"
echo "Creating user: $DB_USER"
echo ""

# Create database and user
psql postgres << EOF
-- Drop if exists (for fresh start)
DROP DATABASE IF EXISTS $DB_NAME;
DROP USER IF EXISTS $DB_USER;

-- Create database and user
CREATE DATABASE $DB_NAME;
CREATE USER $DB_USER WITH ENCRYPTED PASSWORD '$DB_PASSWORD';
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;

-- Connect and grant schema permissions
\c $DB_NAME
GRANT ALL ON SCHEMA public TO $DB_USER;
GRANT ALL ON ALL TABLES IN SCHEMA public TO $DB_USER;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO $DB_USER;

\echo '✅ Database created successfully!'
\echo ''
\echo 'Connection details:'
\echo '  Database: $DB_NAME'
\echo '  User: $DB_USER'
\echo '  Password: $DB_PASSWORD'
\echo ''
\echo 'Add to your .env file:'
\echo 'DATABASE_URL=postgresql://$DB_USER:$DB_PASSWORD@localhost:5432/$DB_NAME'
EOF

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Add DATABASE_URL to your .env file:"
echo "   DATABASE_URL=postgresql://$DB_USER:$DB_PASSWORD@localhost:5432/$DB_NAME"
echo ""
echo "2. Run the app:"
echo "   streamlit run streamlit_app.py"
echo ""
