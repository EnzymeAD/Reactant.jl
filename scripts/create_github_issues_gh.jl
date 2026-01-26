#!/usr/bin/env julia
"""
Script to create GitHub issues from proposed_issues.md using GitHub CLI.

This script:
1. Parses scripts/proposed_issues.md
2. Extracts each proposed issue with its TODO list
3. Creates GitHub issues via `gh` CLI
4. Saves the mapping to scripts/created_issues.csv

Prerequisites:
    - GitHub CLI installed (https://cli.github.com/)
    - Authenticated with: gh auth login
    - Generated: scripts/proposed_issues.md

Usage:
    julia scripts/create_github_issues_gh.jl
"""

function parse_proposed_issues(md_file::String)
    issues = []
    
    content = read(md_file, String)
    
    # Split by "Proposed Issue N:" headers
    pattern = r"## Proposed Issue \d+: (.+?)\n\n\*\*Count\*\*: (\d+) TODOs\n\n### Issue Description\n\n(.+?)\n\n### TODO Items\n\n(.+?)(?=\n---|\z)"s
    
    for m in eachmatch(pattern, content)
        category = strip(m[1])
        count = parse(Int, m[2])
        description = strip(m[3])
        todo_items = strip(m[4])
        
        # Format the issue body
        body = """
        $description
        
        ## TODO Items
        
        $todo_items
        
        ---
        
        **Related:** #2224 (Meta-issue for TODO tracking)
        **Category:** $category
        **Count:** $count items
        """
        
        push!(issues, (category=category, body=body, count=count))
    end
    
    return issues
end

function create_github_issue(repo::String, category::String, body::String)
    title = "[TODO Tracking] $category"
    
    # Create temporary file for body (to handle multiline content)
    body_file = tempname()
    write(body_file, body)
    
    try
        # Use gh CLI to create the issue
        cmd = `gh issue create --repo $repo --title $title --label TODO,tracking --body-file $body_file`
        output = read(cmd, String)
        
        # Extract issue number from URL
        issue_url = strip(output)
        m = match(r"/(\d+)$", issue_url)
        if m !== nothing
            issue_number = parse(Int, m[1])
            return (url=issue_url, number=issue_number)
        else
            error("Could not extract issue number from: $issue_url")
        end
    finally
        rm(body_file; force=true)
    end
end

function main()
    root_dir = dirname(dirname(@__FILE__))
    scripts_dir = joinpath(root_dir, "scripts")
    
    # Check prerequisites
    proposed_file = joinpath(scripts_dir, "proposed_issues.md")
    if !isfile(proposed_file)
        error("Proposed issues file not found: $proposed_file\nRun: julia scripts/categorize_todos.jl")
    end
    
    # Check if gh CLI is available
    if !success(`gh --version`)
        error("GitHub CLI (gh) not found. Install from: https://cli.github.com/")
    end
    
    # Check if authenticated
    if !success(`gh auth status`)
        error("Not authenticated with GitHub CLI. Run: gh auth login")
    end
    
    repo = "EnzymeAD/Reactant.jl"
    
    println("Parsing proposed issues...")
    issues = parse_proposed_issues(proposed_file)
    println("Found $(length(issues)) proposed issues\n")
    
    # Create output file
    output_file = joinpath(scripts_dir, "created_issues.csv")
    open(output_file, "w") do io
        println(io, "category,issue_number,issue_url")
    end
    
    created = 0
    for issue in issues
        println("Creating issue: [TODO Tracking] $(issue.category)")
        println("  Count: $(issue.count) items")
        
        try
            result = create_github_issue(repo, issue.category, issue.body)
            println("  ✓ Created: $(result.url)")
            println("  Issue #$(result.number)")
            
            # Append to CSV
            open(output_file, "a") do io
                println(io, "\"$(issue.category)\",$(result.number),$(result.url)")
            end
            
            created += 1
            
            # Be nice to the API
            sleep(2)
        catch e
            println("  ✗ Error: $e")
            @warn "Failed to create issue for $(issue.category)" exception=e
        end
        
        println()
    end
    
    println("Summary:")
    println("  Created: $created/$(length(issues)) issues")
    println("  Output: $output_file")
    
    if created > 0
        println("\nNext steps:")
        println("  1. Review created issues on GitHub")
        println("  2. Run: julia scripts/map_todos_to_issues.jl")
        println("  3. Run: julia scripts/update_todos.jl scripts/todo_mapping.csv")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
