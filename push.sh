#!/usr/bin/bash

# Check if inside a Git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
  echo "Not a git repository!"
  exit 1
fi

# Parse arguments
if [ $# -eq 0 ]; then
  echo "Usage: $0 [--yes] \"commit message\""
  exit 1
fi

# Default values
skip_confirm=false
commit_msg=""

# Check for --yes and commit message
if [ "$1" == "-y" ]; then
  skip_confirm=true
  shift
fi

# Remaining args are the commit message
commit_msg="$*"

if [ -z "$commit_msg" ]; then
  echo "Commit message required!"
  echo "Usage: $0 [--yes] \"commit message\""
  exit 1
fi

# Get current branch
branch=$(git symbolic-ref --short -q HEAD || git rev-parse --short HEAD)

# Show git status
echo "Git status:"
git status
echo

# Confirm before pushing (unless --yes was passed)
if [ "$skip_confirm" = false ]; then
  echo "Branch: $branch"
  echo "Commit message: \"$commit_msg\""
  read -p "Do you want to commit and push? (y/n): " confirm
  if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborting push."
    exit 0
  fi
fi

# Stage, commit, and push
echo "Adding changes..."
git add .

echo "Committing..."
if git commit -m "$commit_msg"; then
  echo "Pushing to origin/$branch..."
  if git push origin "$branch"; then
    echo "Push successful"
  else
    echo "Push failed"
    exit 1
  fi
else
  echo "Commit failed (maybe nothing to commit?)"
  exit 1
fi