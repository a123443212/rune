# Failed: cross-language node-count equality

Attempt: require identical C++ and Rust node counts on the same position and depth as a parity gate.

Result: rejected as a gate. The two harnesses use different move generators and orderings, so counts differ (266 versus 98 nodes at depth 2 on startpos with agreeing leaf evals). Parity is defined at leaf evaluations plus per-language determinism plus first-divergence tracing, not at raw node counts. Forcing equal counts would couple the test to movegen trivia.
