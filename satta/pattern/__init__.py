"""Pattern Engine and Engine 5: the upgraded levels, running beside engine 4.0 with their own files.

They never write engine 4.0's files (data/predictions.jsonl, data/dashboard.json); they only
read the shared results. Each keeps its own locked predictions, history and dashboard
(data/pattern/, data/engine5/). A failure here cannot stop engine 4.0 (separate watcher steps).
"""

ENGINE = "P-1.0"
ENGINE5 = "5.0"
