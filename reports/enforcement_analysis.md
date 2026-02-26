    # Enforcement Action Effectiveness Analysis

    ## 1. Pre / Post Window Analysis  (±100 μs)

    ### 1.1  Per Action Type
    | action_type | n_events | pre_mean | post_mean | delta_mean | delta_mean_std | delta_p95 | mean_restoration_us |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Access Control | 150 | 99.6301 | 99.9331 | 0.3030 | 2.1912 | 0.6626 | 7.8911 |
| Isolation | 193 | 99.8671 | 99.8667 | -0.0004 | 2.2401 | -0.0468 | 8.1882 |
| Traffic Redirection | 162 | 100.0522 | 99.8367 | -0.2155 | 2.0700 | -0.2452 | 7.2017 |

    ### 1.2  Per Action Type × Attack Type
    | action_type | attack_type | n | delta_mean | delta_p95 |
| --- | --- | --- | --- | --- |
| Access Control | Data Injection | 30 | 0.5606 | 0.8302 |
| Access Control | DoS | 30 | 0.6045 | 2.1055 |
| Access Control | MITM | 23 | 0.1622 | 0.1867 |
| Access Control | Replay | 28 | -0.1231 | 0.1074 |
| Access Control | Spoofing | 39 | 0.2619 | 0.1028 |
| Isolation | Data Injection | 40 | 0.1084 | -0.7592 |
| Isolation | DoS | 40 | 0.4387 | 0.7706 |
| Isolation | MITM | 30 | -0.2796 | 0.5841 |
| Isolation | Replay | 59 | -0.2055 | -0.1970 |
| Isolation | Spoofing | 24 | -0.0599 | -0.6416 |
| Traffic Redirection | Data Injection | 31 | -0.0107 | 0.0416 |
| Traffic Redirection | DoS | 40 | -0.3350 | -0.6582 |
| Traffic Redirection | MITM | 29 | 0.3377 | -0.2784 |
| Traffic Redirection | Replay | 26 | -0.1545 | 0.5379 |
| Traffic Redirection | Spoofing | 36 | -0.7488 | -0.5722 |

    ## 2. Difference-in-Differences (DiD)

    | method | action_type | ate_estimate | ci_lower | ci_upper | p_value | n_treated | n_control |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DiD | Access Control | 0.1916 | -0.2663 | 0.6601 | 0.4233 | 150 | 150 |
| DiD | Isolation | 0.3829 | -0.0804 | 0.8230 | 0.0926 | 193 | 193 |
| DiD | Traffic Redirection | -0.2514 | -0.7202 | 0.2127 | 0.3038 | 162 | 162 |

    **Interpretation:** A negative ATE means enforcement *reduced* latency
    relative to the control window; a positive value means transient overhead.

    ## 3. Propensity-Score Matching (PSM)

    | method | action_type | ate_estimate | ci_lower | ci_upper | p_value | n_treated | n_control |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PSM | Access Control | -8.4320 | -10.8136 | -5.9642 | 0.0000 | 150 | 150 |
| PSM | Isolation | -7.1940 | -9.3235 | -5.0433 | 0.0000 | 193 | 193 |
| PSM | Traffic Redirection | -8.5856 | -10.7175 | -6.3865 | 0.0000 | 162 | 162 |

    ## 4. Combined ATE Summary

    | method | action_type | ate_estimate | ci_lower | ci_upper | p_value | n_treated | n_control |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DiD | Access Control | 0.1916 | -0.2663 | 0.6601 | 0.4233 | 150 | 150 |
| DiD | Isolation | 0.3829 | -0.0804 | 0.8230 | 0.0926 | 193 | 193 |
| DiD | Traffic Redirection | -0.2514 | -0.7202 | 0.2127 | 0.3038 | 162 | 162 |
| PSM | Access Control | -8.4320 | -10.8136 | -5.9642 | 0.0000 | 150 | 150 |
| PSM | Isolation | -7.1940 | -9.3235 | -5.0433 | 0.0000 | 193 | 193 |
| PSM | Traffic Redirection | -8.5856 | -10.7175 | -6.3865 | 0.0000 | 162 | 162 |

    ## 5. Recommendations

    ### Best Action per Attack Type  (lowest Δ mean latency)
    | attack_type | recommended_action | delta_mean | delta_p95 |
| --- | --- | --- | --- |
| Data Injection | Traffic Redirection | -0.0107 | 0.0416 |
| DoS | Traffic Redirection | -0.3350 | -0.6582 |
| MITM | Isolation | -0.2796 | 0.5841 |
| Replay | Isolation | -0.2055 | -0.1970 |
| Spoofing | Traffic Redirection | -0.7488 | -0.5722 |

    ### Best Action per Severity Level
    | severity_level | recommended_action | delta_mean |
| --- | --- | --- |
| Critical | Traffic Redirection | -0.3758 |
| High | Traffic Redirection | 0.2359 |
| Low | Isolation | -0.8774 |
| Medium | Traffic Redirection | -0.6845 |

    ## 6. Methodology Notes

    | Step | Detail |
    | --- | --- |
    | Window | ±100 μs around each enforcement timestamp |
    | DiD control | Same-size window shifted 5× away from treatment event |
    | PSM covariates | cpu_usage, memory_usage, queue_occupancy, trust_score, packet_size_bytes, flow_priority, traffic_deviation, behavior_anomaly_score, queue_delay_us |
    | CI method | Percentile bootstrap (2000 resamples, α=0.05) |
    | p-value | Welch's two-sample t-test |

    > **Caveat:** The dataset is *synthetic* with near-uniform latency, so large
    > causal effects are not expected.  The methodology provides a sound framework
    > that transfers directly to production data.
