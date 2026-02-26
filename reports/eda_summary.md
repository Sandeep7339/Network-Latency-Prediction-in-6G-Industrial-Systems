# EDA Summary — Combined_Dataset_40k

**Dataset:** 40,000 rows × 38 columns (stratified sample from 200k-row spine)  
**Missing Values:** 0 across all columns  
**Generated:** February 2026

---

## Key Observations

1. **Zero missingness.** All 38 columns are fully populated after the merge pipeline. The merge-asof with 200ms tolerance and left-joins produced complete coverage — no imputation will be needed.

2. **Latency is uniformly distributed.** `latency_us` ranges from 41.0 to 159.9 µs with a nearly flat histogram (mean ≈ median ≈ 100 µs). This suggests the source data is synthetically generated from a uniform distribution rather than a realistic heavy-tailed latency profile. Log-scale histogram confirms no tail structure.

3. **Jitter is also uniformly distributed.** `jitter_us` spans 0.10–14.16 µs (mean ≈ 5.01, median ≈ 5.00). Like latency, it shows a uniform profile with no spikes or outliers.

4. **Traffic types are balanced.** The four traffic types — Command (10,047), Control (10,024), Video (10,001), Sensor (9,928) — are nearly perfectly balanced, confirming the stratified sampling worked correctly. Packet sizes are uniformly distributed (64–1,499 bytes) and identical across traffic types (boxplots overlap completely).

5. **All rows are flagged as anomalous.** `anomaly_label` has only 1 unique value (all True). This column is **not useful as a target or feature** — it carries no discriminative information. If the goal is anomaly detection, the label must be sourced differently or the raw Security_Events should be re-examined.

6. **Enforcement success rate is high and uniform across severity levels.** Overall success rate is 95.1%. The rate is essentially identical for Low, Medium, High, and Critical severity (all ~95%). This means `severity_level` alone does not predict enforcement failure.

7. **Near-zero correlations between features.** The top absolute correlation with `latency_us` (excluding its ms-scaled copy) is merely |r| = 0.010 (`battery_level`, `cpu_usage`). All numeric features appear independently generated, with no meaningful linear relationships. The pair plot confirms this: scatter panels show uniform 2D clouds with no structure.

8. **Controller state is mostly Normal.** 69.8% of rows are "Normal", 20.1% "Congested", 10.1% "Under Attack". Queue occupancy is surprisingly similar across all three states (~52.4–52.9 mean), which weakens `controller_state` as a meaningful predictor unless interaction effects are present.

9. **Device profiles are shared.** 1,000 unique devices appear, drawn from 4 types (PLC 28%, Sensor 26%, Robot 23%, Actuator 23%) and 4 vendors. Since Device_Profile was left-joined on `src_device_id`, many spine rows share the same device attributes (e.g., `cpu_usage`, `trust_score`), making these columns low-cardinality in practice.

10. **Attack types are evenly distributed.** Five attack types (Data Injection, Replay, DoS, MITM, Spoofing) each account for ~15–21% of rows. Combined with observation #5, every row in the dataset is simultaneously labeled as anomalous and associated with an attack — there are no "clean" baseline rows.

11. **Temporal structure is sparse.** Timestamps span a tiny range (~0.2 s in nanosecond epoch). Per-device time series have very few data points (the most active device has only 61 rows), limiting meaningful time-series analysis or temporal feature engineering.

12. **Mobility and operational state show moderate imbalance.** Mobile devices outnumber Static ones 53:47. Operational state is 79% Normal, 17% Degraded, 4% Fault — a moderate class imbalance that may require stratification if used as a target.

---

## Recommended Feature List for Modeling

Based on the EDA, the following features are recommended as starting inputs. Features with no observed variance or perfect redundancy are excluded.

### Numeric Features

| Feature                   | Rationale                        |
| ------------------------- | -------------------------------- |
| `latency_us`              | Core network performance metric  |
| `jitter_us`               | Transmission quality indicator   |
| `packet_size_bytes`       | Traffic payload characteristic   |
| `flow_priority`           | Scheduling priority (1–8)        |
| `scheduled_slot`          | Time-slot allocation (0–126)     |
| `cpu_usage`               | Device load indicator            |
| `memory_usage`            | Device resource pressure         |
| `battery_level`           | Mobile device health             |
| `trust_score`             | Device trustworthiness (0.4–1.0) |
| `traffic_deviation`       | Anomaly magnitude from baseline  |
| `behavior_anomaly_score`  | ML-derived anomaly indicator     |
| `enforcement_latency_us`  | Response time of security action |
| `control_action_delay_us` | Controller response time         |
| `queue_occupancy`         | Network congestion proxy         |
| `rerouted_flows`          | Flow rerouting activity          |
| `affected_flows`          | Blast radius of enforcement      |

### Categorical Features (encode via one-hot or ordinal)

| Feature             | Cardinality | Encoding                                 |
| ------------------- | ----------- | ---------------------------------------- |
| `traffic_type`      | 4           | One-hot                                  |
| `protocol`          | 3           | One-hot                                  |
| `device_type`       | 4           | One-hot                                  |
| `vendor`            | 4           | One-hot                                  |
| `mobility_state`    | 2           | Binary                                   |
| `operational_state` | 3           | Ordinal (Normal < Degraded < Fault)      |
| `attack_type`       | 5           | One-hot                                  |
| `severity_level`    | 4           | Ordinal (Low < Medium < High < Critical) |
| `action_type`       | 3           | One-hot                                  |
| `controller_state`  | 3           | Ordinal                                  |

### Features to **Exclude**

| Feature                          | Reason                                                  |
| -------------------------------- | ------------------------------------------------------- |
| `anomaly_label`                  | Single unique value (all True) — zero variance          |
| `timestamp_ns`                   | Raw nanosecond epoch — derive temporal features instead |
| `event_id`, `action_id`          | Row identifiers, no predictive value                    |
| `src_device_id`, `dst_device_id` | High cardinality IDs — use device attributes instead    |
| `firmware_version`               | Only 3 values, low signal                               |
| `packet_loss`                    | Binary (0/1), extremely low variance to check           |
| `latency_restored`               | Binary (0/1), closely tied to controller action         |
| `scheduling_adjustment`          | Very low cardinality (11 values)                        |

### Derived Features to Consider

- **latency_jitter_ratio**: `latency_us / jitter_us` — captures relative variability
- **device_load_composite**: `cpu_usage * memory_usage` or PCA combination
- **is_high_severity**: binary flag for Critical + High severity
- **is_under_attack**: binary from `controller_state == "Under Attack"`
- **packet_rate_per_device**: count of rows per device (requires group-by)

---

## Figures

| #   | File                                       | Description                             |
| --- | ------------------------------------------ | --------------------------------------- |
| 1   | `figures/01_missingness_heatmap.png`       | Missingness check (all clean)           |
| 2   | `figures/02_latency_us_hist.png`           | Latency distribution (linear + log)     |
| 3   | `figures/03_jitter_us_hist.png`            | Jitter distribution                     |
| 4   | `figures/04_packet_size_dist.png`          | Packet size histogram + boxplot by type |
| 5   | `figures/05_latency_vs_enforcement.png`    | Latency vs enforcement scatter          |
| 6   | `figures/06_correlation_heatmap.png`       | Full correlation matrix                 |
| 7   | `figures/07_device_timeline.png`           | Per-device 1-s resampled timeline       |
| 8   | `figures/08_attack_timeline.png`           | Attack event frequency timeline         |
| 9   | `figures/09_latency_by_traffic_type.png`   | Violin plot by traffic type             |
| 10  | `figures/10_queue_by_controller_state.png` | Queue occupancy by state                |
| 11  | `figures/11_success_rate_by_severity.png`  | Enforcement success rates               |
| 12  | `figures/12_pair_plot.png`                 | Key metrics pair plot                   |
