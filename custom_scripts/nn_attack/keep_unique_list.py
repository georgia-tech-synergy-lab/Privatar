f = open("/home/jianming/work/Privatar_prj/custom_scripts/nn_attack/selected_expression_frame_list.txt", "r")
lines = f.readlines()

# Initialize a set to track the first fields we've seen
seen_first_fields = set()

# Initialize a list to store the unique lines
unique_lines = []

# Iterate over the lines
for line in lines:
    first_field = line.split()[0]  # Get the first field
    if first_field not in seen_first_fields:
        unique_lines.append(line)  # Keep the line
        seen_first_fields.add(first_field)  # Mark the first field as seen

# Print the unique lines
for unique_line in unique_lines:
    print(unique_line,end="")