# truncate datasets
python3 -m decalfc.scripts.enforce_dates --kwargs \
    incubator="eclipse" \
    dates="./ref/eclipse_incubation_times.json" \
    versions='{"tech": "1", "social": "1"}' \
    save_versions='{"tech": "2", "social": "2"}'

# re-infer replies
python3 -m decalfc.scripts.pre_process --kwargs \
    incubator="eclipse" \
    load_versions='{"tech": "2", "social": "2"}' \
    save_versions='{"tech": "3", "social": "3"}'

