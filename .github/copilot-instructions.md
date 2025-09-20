# XSI LLMs Attack Repository

**FOLLOW THESE INSTRUCTIONS COMPLETELY.** Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Repository Status

**CRITICAL**: This repository is currently minimal and contains only a README.md file. **DO NOT** attempt to build, test, or run anything as there are no build scripts or source code files present.

## Working Effectively

### Current Repository State
- **VERIFY**: Repository contains only: `README.md` with title "# xsi-llms-attack"
- **CONFIRM**: No source code files exist yet
- **CHECK**: No build configuration (package.json, requirements.txt, etc.)
- **UNDERSTAND**: No dependencies to install
- **REALIZE**: No build or test scripts exist
- **KNOW**: No CI/CD workflows configured

### Repository Structure
```
xsi-llms-attack/
├── .github/
│   └── copilot-instructions.md  # This file
├── README.md          # Project title only
└── .git/             # Git repository metadata
```

### Validated Commands
**USE THESE COMMANDS** - The following commands have been validated to work in the current repository state:

#### Basic Repository Operations - ALWAYS WORK
- **RUN**: `git status` -- shows working tree status (may show untracked .github/ directory)
- **RUN**: `git log --oneline` -- shows commit history
- **RUN**: `ls -la` -- lists repository contents
- **RUN**: `cat README.md` -- displays: "# xsi-llms-attack"

#### File Operations That Work
- **USE**: `mkdir <directory>` -- create new directories
- **USE**: `touch <filename>` -- create new files
- **USE**: Standard file editing operations

#### Commands That Currently Fail - DO NOT USE
- **AVOID**: Any build commands (no build system exists)
- **AVOID**: Any test commands (no tests exist)
- **AVOID**: Any dependency installation (no dependency files exist)
  - **FAILS**: `npm install` fails with "Could not read package.json" (may create empty package-lock.json)
  - **FAILS**: `pip install -r requirements.txt` fails with "No such file or directory"
- **AVOID**: Any application startup (no application exists)
- **FAILS**: `make` fails with "No targets specified and no makefile found"

## Development Setup

### For Future Development
**WHEN** this repository is populated with actual code, **FOLLOW** this typical setup:

1. **IF Python-based project**: Look for `requirements.txt`, `setup.py`, or `pyproject.toml`
2. **IF Node.js project**: Look for `package.json` and **RUN** `npm install`
3. **IF Research notebooks**: Look for Jupyter notebooks (`.ipynb` files)
4. **IF ML/AI project**: Look for `conda.yml` or `environment.yml` for environment setup

### Environment Preparation
**BEFORE** adding code to this repository:
- **ENSURE** appropriate Python environment (typically 3.8+)
- **CONSIDER** using virtual environments (`python -m venv venv`)
- **DOCUMENT** any specific requirements in README.md

## Validation Requirements

### Current Validation Steps
**SINCE** no code exists, **PERFORM** validation by:
- **VERIFY** repository structure with `ls -la`
- **CONFIRM** git status with `git status`
- **CHECK** README content with `cat README.md`

### Future Validation Steps
**WHEN** code is added, **ALWAYS**:
- **TEST** any installation steps you document
- **VERIFY** build processes work end-to-end
- **RUN** test suites if they exist
- **VALIDATE** any example usage scenarios
- **TEST** on a fresh clone to ensure reproducibility

## Common Tasks

### Adding New Code
**WHEN** adding the first code files:
1. **CREATE** appropriate directory structure
2. **ADD** dependency management files (requirements.txt, package.json, etc.)
3. **UPDATE** README.md with proper documentation
4. **ADD** build scripts if needed
5. **INCLUDE** tests from the beginning
6. **SET UP** CI/CD workflows in `.github/workflows/`

### Repository Information
- **Project Type**: LLM Attack/Security Research (inferred from name)
- **Current State**: Empty repository placeholder
- **Next Steps**: Populate with actual research code and documentation

### Timing Expectations
- Repository exploration: < 1 minute
- File operations: < 1 minute
- Future builds: TBD when build system is added
- Future tests: TBD when tests are added

## Important Notes

- **DO NOT** attempt to run build commands - they will fail
- **DO NOT** try to install dependencies - none are defined
- **DO NOT** look for source code files - they don't exist yet
- **Always** check the current repository state before assuming functionality exists
- When populating this repository, follow security research best practices
- Consider ethical implications of LLM attack research and ensure responsible disclosure

## Research Context

Based on the repository name "xsi-llms-attack", this appears to be intended for:
- Large Language Model security research
- Attack vector analysis
- Possibly Cross-Site Injection (XSI) techniques applied to LLMs

Ensure any future development follows responsible disclosure practices and academic research ethics.