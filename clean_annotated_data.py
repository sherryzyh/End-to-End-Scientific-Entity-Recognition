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
            data = line.split(" ")
            if len(data) == 0 or line.strip() == "":
                continue
            if len(data) != 2:
                not_fixed += 1
                print(f"not fixed - line {i}: {lines[i]}")
                continue
            data[1] = data[1].strip()
            closest = difflib.get_close_matches(data[1], label_types, 1)
            if not closest:
                print(f"before - line {i}: {lines[i]}")
                lines[i] = " ".join([data[0], "O"])
                prev_label = "O"
                print(f"after: {lines[i]}")
                fixed += 1
                continue
            label = closest[0]
            if label == "O":
                prev_label = label
                continue
            if label == prev_label:
                if data[1] != "I-" + label:
                    print(f"before - line {i}: {lines[i]}")
                    lines[i] = " ".join([data[0], "I-" + label])
                    print(f"after: {lines[i]}")
                    fixed += 1
            else:
                if data[1] != "B-" + label:
                    print(f"before - line {i}: {lines[i]}")
                    lines[i] = " ".join([data[0], "B-" + label])
                    print(f"after: {lines[i]}")
                    fixed += 1
                prev_label = label
    print(f"{filename}: {fixed} errors have been fixed; {not_fixed} errors remain")
    return lines
            
if __name__ == "__main__":
    source = 'Annotated_Data/annotated_paper'
    destination = 'cleaned_data'

    for filename in os.listdir(source):
        print(filename)
        lines = clean(source, filename)
        with open(os.path.join(destination, "cleaned_" + filename), 'w', encoding="utf-8") as f:
            f.write("\n".join(lines))