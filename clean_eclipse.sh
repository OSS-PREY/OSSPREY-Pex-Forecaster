# truncate datasets
python3 -m decalfc.scripts.enforce_dates --kwargs \
    incubator="eclipse" \
    dates="./ref/eclipse_incubation_times.json" \
    versions='{"tech": "1", "social": "1"}' \
    save_versions='{"tech": "2", "social": "2"}'

