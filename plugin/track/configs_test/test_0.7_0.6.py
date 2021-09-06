_base_ = [
    './test.py'
]

model = dict(
    score_thresh=0.7,
    filter_score_thresh=0.6,    
)
