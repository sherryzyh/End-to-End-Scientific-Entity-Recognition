import os
from collections import defaultdict

class AnnotationAnalyzer:
    def __init__(self, annotation_root="Annotated_Data/annotated_paper"):
        self.annotation_root = annotation_root
        self.total_sent_n = 0
        self.entity_sent_stat = defaultdict(lambda: 0)
        self.entity_stat = defaultdict(lambda: 0)

    def analyze(self):
        total_sent_n = 0
        entity_sent_stat = defaultdict(lambda: 0)
        entity_stat = defaultdict(lambda: 0)

        paper_list = os.listdir(self.annotation_root)
        for paper in paper_list:
            read_path = os.path.join(self.annotation_root, paper)
            with open(read_path, "r", encoding='utf-8') as f:
                lines = f.read().splitlines()

            currsentence = ""
            curr_entity_set = set()
            for line in lines:

                if len(line) == 0 or len(line.strip().split(" ")) < 2:
                    if len(currsentence) > 0:
                        total_sent_n += 1
                        if len(curr_entity_set) > 0:
                            entity_sent_stat["any"] += 1
                            for e in curr_entity_set:
                                entity_sent_stat[e] += 1
                    currsentence = ""
                    curr_entity_set = set()
                    continue

                token, label = line.strip().split(" ")
                currsentence = currsentence + " " + token
                # print(paper, line)
                if label[0] == "B":
                    curr_entity_set.add(label[2:])
                    entity_stat[label[2:]] += 1

        self.total_sent_n = total_sent_n
        self.entity_sent_stat = entity_sent_stat
        self.entity_stat = entity_stat

    def display(self):
        print(f"There are {self.total_sent_n} sentences in total.")
        print("The number of sentences containing entity ...")
        for e, c in self.entity_sent_stat.items():
            print(f"Entity {e:20}#Sent {c}\tproportion {c/self.total_sent_n:.2f}")
        print("The number of entities' occurrence ...")
        for e, c in self.entity_stat.items():
            print(f"Entity {e:20}#Occurrence {c}")


if __name__=="__main__":
    project_root = os.getcwd()
    annotation_root = os.path.join(project_root, "Annotated_Data", "annotated_paper")
    analyzer = AnnotationAnalyzer(annotation_root)
    analyzer.analyze()
    analyzer.display()