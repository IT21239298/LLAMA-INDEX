def create_gitignore():
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/

# Virtual Environment
venv/
env/
.env/

# IDE
.vscode/
.idea/

# Jupyter Notebook
.ipynb_checkpoints

# Local development settings
.env
.env.local

# Logs
*.log
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)

# Run the function
if __name__ == "__main__":
    create_gitignore()