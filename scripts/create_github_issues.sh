#!/bin/bash
# Wrapper script to create GitHub issues for TODO tracking using GitHub CLI
#
# This is a simple wrapper around create_github_issues_gh.jl
# 
# Prerequisites:
#   - GitHub CLI installed (https://cli.github.com/)
#   - Authenticated with: gh auth login
#   - Generated files: scripts/proposed_issues.md
#
# Usage:
#   bash scripts/create_github_issues.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the Julia script that does the actual work
julia "$SCRIPT_DIR/create_github_issues_gh.jl"
