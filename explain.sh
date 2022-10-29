for system in `ls /work/yinghuan/projects/End-to-End-Scientific-Entity-Recognition/predictions`; do
    echo "*****" ${system} "*****"

    python tool/compare_groundtruth_and_prediction.py -i ${system}

    python -m explainaboard_client.cli.evaluate_system \
    --username yinghuan@andrew.cmu.edu \
    --api-key pzG2G_8ZYgLP1jdsINqC-w \
    --task named-entity-recognition \
    --system-name jyy_baseline_bert_train_unweighted_everysent \
    --custom-dataset-file /work/yinghuan/projects/End-to-End-Scientific-Entity-Recognition/Dataset/val_data_ground_truth/output_validation_data_ground_truth.conll \
    --custom-dataset-file-type conll  \
    --system-output-file /work/yinghuan/projects/End-to-End-Scientific-Entity-Recognition/predictions/$system/output_$system.conll \
    --system-output-file-type conll \
    --source-language en
done

# system="baseline_bert_train_unweighted_everysent"
# echo "*****" ${system} "*****"

# python -m explainaboard_client.cli.evaluate_system \
#   --username yinghuan@andrew.cmu.edu \
#   --api-key pzG2G_8ZYgLP1jdsINqC-w \
#   --task named-entity-recognition \
#   --system-name jyy_baseline_bert_train_unweighted_everysent \
#   --custom-dataset-file /work/yinghuan/projects/End-to-End-Scientific-Entity-Recognition/Dataset/val_data_ground_truth/output_validation_data_ground_truth.conll \
#   --custom-dataset-file-type conll  \
#   --system-output-file /work/yinghuan/projects/End-to-End-Scientific-Entity-Recognition/predictions/$system/output_$system.conll \
#   --system-output-file-type conll \
#   --source-language en