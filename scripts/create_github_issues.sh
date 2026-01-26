#!/bin/bash
# Script to create GitHub issues for TODO tracking using GitHub CLI (gh)
# 
# Prerequisites:
#   - GitHub CLI installed (https://cli.github.com/)
#   - Authenticated with: gh auth login
#   - Generated files: scripts/proposed_issues.md
#
# Usage:
#   bash scripts/create_github_issues.sh

set -e

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed."
    echo "Install from: https://cli.github.com/"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "Error: Not authenticated with GitHub CLI."
    echo "Run: gh auth login"
    exit 1
fi

# Repository
REPO="EnzymeAD/Reactant.jl"

echo "Creating TODO tracking issues in $REPO..."
echo ""

# Define the categories and their issue bodies
# These would need to be extracted from proposed_issues.md
# For now, this is a template showing the structure

declare -a CATEGORIES=(
    "General Improvements"
    "Error Handling & Validation"
    "Sharding & Multi-Device Support"
    "MLIR/XLA Integration"
    "Build System"
    "Type System"
    "Array Operations"
    "Test Coverage"
    "Performance Optimization"
    "GPU/TPU Support"
    "Documentation"
    "Extension Support"
    "Linear Algebra"
)

# This function would need to extract the TODO list for each category from proposed_issues.md
create_issue() {
    local category="$1"
    local title="[TODO Tracking] $category"
    
    echo "Creating issue: $title"
    
    # Note: In practice, you'd need to extract the body from proposed_issues.md
    # This is a placeholder showing the structure
    local body="This issue tracks TODO items related to $category.

See scripts/proposed_issues.md for the complete list of TODO items in this category.

Related: #2224"
    
    # Create the issue and capture the issue number
    issue_url=$(gh issue create \
        --repo "$REPO" \
        --title "$title" \
        --body "$body" \
        --label "TODO,tracking")
    
    issue_number=$(echo "$issue_url" | grep -oP '\d+$')
    echo "  Created: $issue_url (Issue #$issue_number)"
    echo "$category,$issue_number" >> scripts/created_issues.csv
}

# Initialize the output file
echo "category,issue_number" > scripts/created_issues.csv

# Create issues for each category
for category in "${CATEGORIES[@]}"; do
    create_issue "$category"
    sleep 1  # Be nice to the API
done

echo ""
echo "✓ Created ${#CATEGORIES[@]} tracking issues"
echo "✓ Issue numbers saved to scripts/created_issues.csv"
echo ""
echo "Next steps:"
echo "1. Review the created issues on GitHub"
echo "2. Run: julia scripts/map_todos_to_issues.jl"
echo "3. Run: julia scripts/update_todos.jl scripts/todo_mapping.csv"
