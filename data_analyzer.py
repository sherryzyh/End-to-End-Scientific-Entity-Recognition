import os
from collections import defaultdict
# TODO: the length statistic of different entities

class AnnotationAnalyzer:
    def __init__(self,
                 annotation_root="Annotated_Data/annotated_paper",
                 stats_file_path="Annotated_Data/stats.txt"):
        self.annotation_root = annotation_root
        self.total_sent_n = 0
        self.total_paper_n = 0
        self.entity_sent_stat = defaultdict(lambda: 0)
        self.entity_stat = defaultdict(lambda: 0)
        self.stats_file_path = stats_file_path

    def analyze(self):
        total_sent_n = 0
        total_paper_n = 0
        entity_sent_stat = defaultdict(lambda: 0)
        entity_stat = defaultdict(lambda: 0)

        paper_list = os.listdir(self.annotation_root)
        for paper in paper_list:
            read_path = os.path.join(self.annotation_root, paper)
            try:
                with open(read_path, "r", encoding='utf-8') as f:
                    lines = f.read().splitlines()
            except:
                print(f"An exception occurred when reading << {paper} >>")
                continue

            currsentence = ""
            curr_entity_set = set()
            for line in lines:

                if len(line) == 0 or len(line.strip().split(" ")) < 2:
                    if len(currsentence) > 0:
                        total_sent_n += 1
                        if len(curr_entity_set) > 0:
                            entity_sent_stat["ANY"] += 1
                            for e in curr_entity_set:
                                entity_sent_stat[e] += 1
                    currsentence = ""
                    curr_entity_set = set()
                    continue

                token, label = line.strip().split(" ")
                currsentence = currsentence + " " + token
                if label[0] == "B":
                    curr_entity_set.add(label[2:])
                    entity_stat[label[2:]] += 1
            total_paper_n += 1

        self.total_sent_n = total_sent_n
        self.total_paper_n = total_paper_n
        self.entity_sent_stat = entity_sent_stat
        self.entity_stat = entity_stat

    def display(self):
        with open(self.stats_file_path, "w", encoding="utf-8") as f:
            f.write(f"There are {self.total_paper_n} papers in total.\n")
            f.write("-" * 40 + "\n")
            f.write(f"There are {self.total_sent_n} sentences in total.\n\n")
            for e, c in self.entity_sent_stat.items():
                f.write(f"Entity {e:20}#Sent {c:6}\tproportion {c / self.total_sent_n:.4f}\n")
            f.write("-" * 40 + "\n")
            total_entities_occurrence = sum(self.entity_stat.values())
            f.write(f"There are {total_entities_occurrence} entities in total.\n\n")
            for e, c in self.entity_stat.items():
                f.write(f"Entity {e:20}#Occur {c:5}\tproportion {c / total_entities_occurrence: .4f}\n")

        print(f"Statistics result is saved to {self.stats_file_path}.")


if __name__=="__main__":
    project_root = os.getcwd()
    annotation_root = os.path.join(project_root, "Annotated_Data", "annotated_paper")
    stats_file_path = os.path.join(project_root, "Annotated_Data", "stats.txt")
    analyzer = AnnotationAnalyzer(annotation_root, stats_file_path)
    analyzer.analyze()
    analyzer.display()