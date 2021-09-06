_base_ = [
    './test.py'
]

model = dict(
    score_thresh=0.5,
    filter_score_thresh=0.4,
)
