## Pipeline Network Generation Script
# python3 -m decalfc.pipeline.pipeline --kwargs \
#     incubator=apache \
#     versions='{"tech": 1, "social": 1}'

# python3 -m decalfc.pipeline.pipeline --kwargs \
#     incubator=github \
#     versions='{"tech": 3, "social": 4}'

python3 -m decalfc.pipeline.pipeline --kwargs \
    incubator=eclipse \
    versions='{"tech": 2, "social": 2}'

# python3 -m decalfc.pipeline.pipeline --kwargs \
#     incubator=osgeo \
#     versions='{"tech": 2, "social": 2}'

# python3 -m decalfc.pipeline.pipeline --kwargs \
#     incubator=ospos \
#     versions='{"tech": 0, "social": 0}'
