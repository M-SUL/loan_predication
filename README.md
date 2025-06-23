# loan_predication
This project aims to help banks predict loan approval

visit the website to try it: https://mhdaliharmalani.pythonanywhere.com/

## For Contributors

### Prerequisites
- Python 3.8+
- Git
- GitHub account with access to this repository

### Getting Started

#### 1. Clone the Repository
```bash
# Clone using SSH (recommended)
git clone git@github.com:M-SUL/loan_predication.git

# OR clone using HTTPS
git clone https://github.com/M-SUL/loan_predication.git

# Navigate to project directory
cd loan_predication
```

#### 2. Set Up Your Development Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run initial database migrations
python manage.py migrate

# Create a superuser (optional, for admin access)
python manage.py createsuperuser
```

#### 3. Verify Setup
```bash
# Run the development server
python manage.py runserver

# Open http://127.0.0.1:8000 in your browser to verify it works
```

### Contributing Workflow

#### 1. Create a Feature Branch
```bash
# Make sure you're on main branch and up to date
git checkout main
git pull origin main

# Create and switch to a new feature branch
git checkout -b feature/your-feature-name
# Example: git checkout -b feature/add-loan-analytics
```

#### 2. Make Your Changes
- Write your code
- Test your changes locally
- Follow the existing code style and structure

#### 3. Commit Your Changes
```bash
# Add files to staging
git add .

# OR add specific files
git add path/to/your/file.py

# Commit with a descriptive message
git commit -m "Add: descriptive message about your changes"

# Examples:
# git commit -m "Add: loan risk analysis feature"
# git commit -m "Fix: bug in loan approval calculation"
# git commit -m "Update: improve UI for loan request form"
```

#### 4. Push Your Branch
```bash
# Push your feature branch to GitHub
git push origin feature/your-feature-name
```

#### 5. Create a Pull Request
1. Go to the GitHub repository: https://github.com/M-SUL/loan_predication
2. Click "Compare & pull request" button
3. Fill out the pull request template:
   - **Title**: Brief description of your changes
   - **Description**: Detailed explanation of what you changed and why
   - **Testing**: Describe how you tested your changes
4. Submit the pull request for review

### Important Notes

#### Authentication Setup
If you encounter permission errors when pushing:

**Option 1: Use SSH (Recommended)**
```bash
# Set remote to use SSH
git remote set-url origin git@github.com:M-SUL/loan_predication.git

# Make sure your SSH key is added to your GitHub account
```

**Option 2: Use Personal Access Token**
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate a new token with `repo` permissions
3. Use the token as your password when prompted

#### Database Migrations
If you make changes to models:
```bash
# Create migration files
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Include migration files in your commit
git add loan_manager/migrations/
```

#### Code Quality
- Write clear, descriptive commit messages
- Test your changes before pushing
- Follow Django best practices
- Keep your branches focused on a single feature/fix

### Project Structure
```
loan_predication/
├── credit_system/          # Django project settings
├── loan_manager/           # Main application
│   ├── models.py          # Database models
│   ├── views.py           # View functions
│   ├── urls.py            # URL patterns
│   ├── templates/         # HTML templates
│   └── ml/                # Machine learning modules
├── manage.py              # Django management script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Getting Help
- Check existing issues and pull requests
- Ask questions in the repository discussions
- Review the Django documentation for framework-specific questions

### Common Commands
```bash
# Start development server
python manage.py runserver

# Run tests
python manage.py test

# Create superuser
python manage.py createsuperuser

# Make migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Check for issues
python manage.py check
```
