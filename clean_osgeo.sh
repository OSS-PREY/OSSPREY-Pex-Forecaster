# # clean column names
# python3 -m decalfc.scripts.enforce_column_names --kwargs \
#     incubator="osgeo" \
#     versions='{"tech": ["0"], "social": ["0i", "0e"]}'

# # standardize sender aliases
# python3 -m decalfc.scripts.standardize_sender_aliases --kwargs \
#     incubator="osgeo" \
#     aliases="./ref/osgeo_aliases.csv" \
#     load_save_versions='{"tech": {"0": "0a"}, "social": {"0i": "0ia", "0e": "0ea"}}'

# combine social data in emails-issues
python3 -m decalfc.scripts.combine_social --kwargs \
    incubator="osgeo" \
    social_versions='["0i", "0e"]' \
    save_version="0"

# truncate datasets
# python3 -m decalfc.scripts.enforce_dates --kwargs \
#     incubator="osgeo" \
#     versions='{"tech": ["0a"], "social": ["0"]}'

