# Separates the input for day 24 by the "inp w" instructions
# and reorders it so commonalities between the code for each digit
# become more apparent
# Lines that are the same for all digits are only stored once.
# If they differ between digits, all possible values are stored.

with open("day24_input.txt") as f:
    data = f.read()

pieces = data.split("inp w")
pieces = [p.strip() for p in pieces]
pieces = [p for p in pieces if p]

possible_lines = {}

for i, partial_program in enumerate(pieces):
    with open(f"day24_input_{i}.txt", "w") as f:
        print(partial_program, file=f)

    for line_nr, line in enumerate(partial_program.splitlines()):
        possible_lines.setdefault(line_nr, list()).append(line)

for line_nr in possible_lines:
    if len(set(possible_lines[line_nr])) == 1:
        possible_lines[line_nr] = possible_lines[line_nr][:1]

import json

# possible_lines = {nr: list(lines) for nr, lines in possible_lines.items()}
with open("day24_possible_lines.json", "w") as f:
    json.dump(possible_lines, f, indent=4)
