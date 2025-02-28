# clean column names
python3 -m decalfc.scripts.enforce_column_names --kwargs \
    incubator="osgeo" \
    versions='{"tech": ["0"], "social": ["0i", "0e"]}'

# standardize sender aliases
python3 -m decalfc.scripts.standardize_sender_aliases --kwargs \
    incubator="osgeo" \
    aliases="./ref/osgeo_aliases.csv" \
    load_save_versions='{"tech": {"0": "0a"}, "social": {"0i": "0ia", "0e": "0ea"}}'

# combine social data in emails-issues
python3 -m decalfc.scripts.combine_social --kwargs \
    incubator="osgeo" \
    social_versions='["0ia", "0ea"]' \
    save_version="0"

# truncate datasets
python3 -m decalfc.scripts.enforce_dates --kwargs \
    incubator="osgeo" \
    dates="./ref/osgeo_incubation_times.json" \
    versions='{"tech": "0a", "social": "0"}' \
    save_versions='{"tech": "1", "social": "1"}'

# regular cleaning steps
python3 -m decalfc.scripts.pre_process --kwargs \
    incubator="osgeo" \
    load_versions='{"tech": "1", "social": "1"}' \
    save_versions='{"tech": "2", "social": "2"}' \

