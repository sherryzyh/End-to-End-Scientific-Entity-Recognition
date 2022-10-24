import os
import argparse
from collections import defaultdict
from utils import EntitySentence
# TODO: the length statistic of different entities

class AnnotationAnalyzer:
    def __init__(self,
                 annotation_root="Annotated_Data/annotated_paper",
                 stats_file_path="Annotated_Data/stats.txt"):
        self.annotation_root = annotation_root
        self.total_paper_n = 0
        self.total_sent_n = 0
        self.avg_sent_len = 0
        self.entity_sent_stat = defaultdict(lambda: 0)
        # self.entity_sent_stat = {"ANY": 0,
        #                          "MethodName": 0,
        #                          "HyperparameterName": 0,
        #                          "HyperparameterValue": 0,
        #                          "MetricName": 0,
        #                          "MetricValue": 0,
        #                          "TaskName": 0,
        #                          "DatasetName": 0}
        self.avg_ensent_len = 0
        self.entity_stat = defaultdict(lambda: 0)
        self.stats_file_path = stats_file_path

    def analyze(self):
        total_sent_n = 0
        total_paper_n = 0
        total_sent_len = 0
        total_ensent_len = 0

        paper_list = os.listdir(self.annotation_root)
        for paper in paper_list:
            read_path = os.path.join(self.annotation_root, paper)
            try:
                with open(read_path, "r", encoding='utf-8') as f:
                    lines = f.read().splitlines()
            except:
                print(f"An exception occurred when reading << {paper} >>")
                continue

            entitysentence = EntitySentence()
            for line in lines:
                entitysentence.readLine(line)
                if entitysentence.isEnd:
                    if len(entitysentence) > 0:
                        # update non-empty sentence
                        total_sent_n += 1
                        total_sent_len += len(entitysentence)

                    if entitysentence.containEntity():
                        # update entity statistics
                        self.entity_sent_stat["ANY"] += 1
                        for entity in entitysentence.entity_set:
                            if entity == "O":
                                continue
                            self.entity_stat[entity] += entitysentence.entityCount(entity)
                            if entitysentence.entityCount(entity) > 0:
                                self.entity_sent_stat[entity] += 1
                        total_ensent_len += len(entitysentence)
                    else:
                        self.entity_sent_stat['NONE'] += 1

                    self.entity_stat['O'] += entitysentence.entityCount('O')

                    entitysentence.clear()


            total_paper_n += 1

        self.total_paper_n = total_paper_n
        self.total_sent_n = total_sent_n
        if total_sent_n > 0:
            self.avg_sent_len = total_sent_len / total_sent_n
        if self.entity_sent_stat["ANY"] > 0:
            self.avg_ensent_len = total_ensent_len / self.entity_sent_stat["ANY"]

    def display(self):
        with open(self.stats_file_path, "w", encoding="utf-8") as f:
            f.write(f"There are {self.total_paper_n} papers in total.\n")

            f.write("-" * 40 + "\n")
            f.write(f"There are {self.total_sent_n} sentences in total.\n\n")
            f.write(f"Avg length of sentence {self.avg_sent_len}.\n\n")
            f.write(f"Avg length of entity sentence {self.avg_ensent_len}.\n\n")

            for e in self.entity_sent_stat.keys():
                f.write(f"Entity {e:25}#Sent {self.entity_sent_stat[e]:6}\tproportion {self.entity_sent_stat[e] / self.total_sent_n:.6f}\n")

            f.write("-" * 40 + "\n")
            total_entities_occurrence = sum(self.entity_stat.values()) - self.entity_sent_stat['O']
            f.write(f"There are {total_entities_occurrence} non-O entities in total.\n\n")

            for e in self.entity_stat.keys():
                f.write(f"Entity {e:25}#Occur {self.entity_stat[e]:5}\tproportion {self.entity_stat[e] / total_entities_occurrence: .6f}\n")

        print(f"Statistics result is saved to {self.stats_file_path}.")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', '-t', action='store_true', help="test")
    args = parser.parse_args()

    project_root = os.getcwd()
    if args.test:
        annotation_root = os.path.join(project_root, "Annotated_Data", "test_annotation")
    else:
        annotation_root = os.path.join(project_root, "Annotated_Data", "cleaned_data")
    stats_file_path = os.path.join(project_root, "Annotated_Data", "stats.txt")
    analyzer = AnnotationAnalyzer(annotation_root, stats_file_path)
    analyzer.analyze()
    analyzer.display()