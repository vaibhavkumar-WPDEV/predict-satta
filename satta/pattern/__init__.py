"""Pattern Engine: the upgraded level, running beside engine 4.0 with its own files.

It never writes engine 4.0's files (data/predictions.jsonl, data/dashboard.json); it only
reads the shared results. Its own locked predictions, history and dashboard live in
data/pattern/. A failure here cannot stop engine 4.0 (separate step in the watcher).
"""

ENGINE = "P-1.0"
