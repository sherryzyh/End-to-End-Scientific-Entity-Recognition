'''
This file cleans human-annotated data in two ways and prints detected issues to console.
1. Typo in label
2. Specifically, any error regarding "B-" vs. "I-"
Note that any line that does not consist of a token and a label is printed to console 
to alert the annotator, but not automatically fixed.
'''

import difflib
import os

label_types = [
    'O', 
    'MethodName', 
    'HyperparameterName', 
    'HyperparameterValue',
    'MetricName',
    'MetricValue',
    'TaskName',
    'DatasetName'
]
    
def clean(source, filename):
    fixed = 0
    not_fixed = 0
    with open(os.path.join(source, filename), 'r', encoding="utf-8") as f:
        lines = f.readlines()
        prev_label = None
        for i, line in enumerate(lines):
            data = line.strip().split(" ")
            if len(data) == 1:
                # print(f"len == 1 | before - line {i}: {lines[i]}")
                if data[0] == "O" or data[0] == "":
                    continue
                else:
                    lines[i] = data[0] + " O\n"
                    fixed += 1
                # print(f"after - line {i}: {lines[i]}")
            elif len(data) >= 3:
                # print(f"len > 3 | before - line {i}: {lines[i]}")
                lines[i] = data[0] + " " + data[-1] + "\n"
                data = lines[i].strip().split(" ")
                fixed += 1

            if len(data) == 2:
                closest = difflib.get_close_matches(data[1], label_types, 1)
                if not closest:
                    print(f"before - line {i}: {lines[i]}")
                    lines[i] = " ".join([data[0], "O", "\n"])
                    prev_label = "O"
                    print(f"after - line {i}: {lines[i]}")
                    fixed += 1
                    continue
                label = closest[0]
                if label == "O":
                    prev_label = label
                    continue
                if label == prev_label:
                    if data[1] != "I-" + label and data[1] != "B-" + label:
                        # print(f"before - line {i}: {lines[i]}")
                        lines[i] = " ".join([data[0], "I-" + label, "\n"])
                        # print(f"after: {lines[i]}")
                        fixed += 1
                else:
                    if data[1] != "B-" + label:
                        # print(f"before - line {i}: {lines[i]}")
                        lines[i] = " ".join([data[0], "B-" + label, "\n"])
                        # print(f"after - line {i}: {lines[i]}")
                        fixed += 1
                    prev_label = label
    print(f"{filename}: {fixed} errors have been fixed; {not_fixed} errors remain")
    return lines
            
if __name__ == "__main__":
    source = 'Annotated_Data/annotated_paper'
    destination = 'Annotated_Data/cleaned_data'

    if not os.path.exists(destination):
        os.mkdir(destination)

    for filename in os.listdir(source):
        print(filename)
        lines = clean(source, filename)
        with open(os.path.join(destination, "cleaned_" + filename), 'w', encoding="utf-8") as f:
            f.write("".join(lines))