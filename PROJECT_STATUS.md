# Project Status

```text
COMMAND_CENTER_DOCS_CREATED
WAITING_FOR_SOURCE_REVIEW
```

Latest command decision:

```text
Use the 2.5-style chain as the main production path.
Use day-ahead and realtime adapters as prediction backends.
Send predictions through residual correction.
Then run learner/fusion.
Then run the negative price classifier.
```

Unresolved inputs:

```text
realtime SOTA source repository/path
2.5 source repository/path
valid model count in day-ahead source repository
2.5 SGDFNet location
P5M canonical pack availability
```
