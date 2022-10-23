from annotated_data_augmenter import AnnotatedDataAug

adg = AnnotatedDataAug()
sentences, sentenceIdx_to_tokenLabelDict, sentenceIdx_to_subNameDict = adg.recover_and_label('./cleaned_data/cleaned_annotated_2015_acl_Addressing the Rare Word Problem in Neural Machine Translation.txt')
print(sentences)
print(sentenceIdx_to_subNameDict)

paraphrased_result_lines = adg.parapharase_and_relabel(sentences, sentenceIdx_to_tokenLabelDict, sentenceIdx_to_subNameDict)
for s in paraphrased_result_lines:
    print(s)