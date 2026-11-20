# Service-only assignment and movement contract

The current environment has four task roles: Search, FOV, COM and Hovering.
Connectivity does not create a task, reserve an aircraft, add a virtual target,
or contribute a movement potential. Every UAV can still forward packets through
the unchanged multi-hop routing layer.

K-KM runs exactly two matching stages at an existing new-RoI assignment
boundary. The first stage contains only discovered FOV tasks. The second
contains only COM tasks and evaluates the same eligible UAV set, including UAVs
that received an FOV task. A UAV can hold at most one FOV and one COM task, and
each service target can appear at most once.

KM runs one combined FOV/COM matching and assigns at most one service task to a
UAV. Random assignment uses `env.assignment_rng` for seeded randomized greedy
selection over feasibility only: round 1 contains FOV, round 2 contains COM,
and a FOV UAV may be reused for COM. `assignment_rounds<=0` runs neither,
`assignment_rounds==1` runs FOV only, and larger values run both typed rounds.
It does not construct or inspect utility values. Search and Hovering remain
fallback roles and do not enter an assignment task set.

Movement observation schema 8 has 619 fields: 17 fields for each of 16 UAVs,
eleven fields for each of eight stable SR-ID slots, the 16 by 16 visited bitmap,
and three global fields. An SR slot becomes valid only after its RoI mission is
known, so dormant manifest SRs cannot reveal future RoIs. Each valid slot holds
current and known-target xyz, direct/support/waiting one-hot state, and
log-normalized FIFO backlog. The task one-hot contains Search, FOV, COM and
Hovering. Movement PBRS contains Search, VS and COM only, each with the formal
default coefficient 3.0. Task-aware ablations mask the complete SR block.

Checkpoint schema 39 records the current service-only, SR-aware observation and
100 m ground-Rayleigh S2G contract. Schema 38 has the older 531-D observation
and is rejected before weights or replay are loaded. Schema 30 used
ROI-center Search discovery, excluded gateway Search contribution and used
mixed Random rounds, so it is rejected before weights or replay load. Schema 29 contains the
retired explicit Relay task, its 595-D movement state and replay fields, and is
rejected before weights or replay are restored. New evaluation and training
runs do not write a dedicated task-role connectivity diagnostics file. Generic
routing, channel and packet diagnostics continue unchanged.
