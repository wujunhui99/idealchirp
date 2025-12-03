#!/usr/bin/env python3
"""Clean requirements.txt by removing conda build paths"""

with open('requirements.txt', 'r') as f:
    lines = f.readlines()

cleaned = []
for line in lines:
    line = line.strip()
    if not line:
        continue

    # If line has ==, it's already clean
    if '==' in line and ' @ file://' not in line:
        cleaned.append(line)
    # If line has @ file://, extract package name
    elif ' @ file://' in line:
        pkg_name = line.split(' @ ')[0].strip()
        # Skip conda-specific packages
        if pkg_name.lower().startswith('conda'):
            continue
        # Add package without version (will get latest)
        cleaned.append(pkg_name)

# Remove duplicates while preserving order
seen = set()
final = []
for line in cleaned:
    if line not in seen:
        seen.add(line)
        final.append(line)

# Write cleaned requirements
with open('requirements_clean.txt', 'w') as f:
    f.write('\n'.join(final) + '\n')

print(f"Cleaned {len(lines)} lines to {len(final)} packages")
print("Saved to requirements_clean.txt")
