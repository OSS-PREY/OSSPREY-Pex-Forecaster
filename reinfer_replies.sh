# re-infer replies for apache
python3 -m decalfc.scripts.pre_process --kwargs \
    incubator="apache" \
    load_versions='{"tech": "1", "social": "1"}' \
    save_versions='{"tech": "2", "social": "2"}'

# # same for github
# python3 -m decalfc.scripts.pre_process --kwargs \
#     incubator="github" \
#     load_versions='{"tech": "3", "social": "4"}' \
#     save_versions='{"tech": "4", "social": "5"}'

