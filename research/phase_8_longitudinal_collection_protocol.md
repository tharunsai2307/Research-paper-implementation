# Phase 8 Longitudinal Collection Protocol & Field Design

## 1. Overview & Research Objective

To establish a scientifically defensible foundation for future temporal disease progression research, this document defines an end-to-end **Longitudinal Field Collection Protocol**.

> [!IMPORTANT]
> The observation intervals, tree IDs, and schemas defined in this document represent **PROPOSED FUTURE PROTOCOLS**, not existing research data. Zero temporal data currently exists in the benchmark dataset.

---

## 2. Field Setup & Specimen Tracking Protocol

```
[Coconut Tree Tagging] 
         ↓ (Weatherproof Barcode / RFID Tag nailed to lower non-functional bark)
[Geospatial Anchoring]
         ↓ (High-precision RTK GNSS coordinates: Latitude, Longitude, Altitude)
[Fixed Observation Station]
         ↓ (Permanent ground marker at 2.5m radius; camera tripod height 1.5m)
[Scheduled Revisit Program]
         ↓ (Day 0 → Day 7 → Day 14 → Day 21 → Day 28)
[Multi-Modal Capture]
         ↓ (RGB Image + Mobile YOLO Inference + Environmental Sensor Telemetry)
[Longitudinal Ingestion Engine]
```

### 2.1 Proposed Observation Program
- **Sampling Frequency**: Every **7 days ($\pm 1$ day)** during high-humidity periods; every **14 days** during dry maintenance periods.
- **Minimum Longitudinal Cohort**:
  - Minimum 50 tracked palm specimens per plantation block.
  - Minimum 5 consecutive revisit intervals per palm ($T_0, T_7, T_{14}, T_{21}, T_{28}$).
- **Observation Uniformity**:
  - Distance: Exactly $2.5\text{ m}$ from trunk.
  - Optical axis: $45^\circ$ elevation angle toward frond canopy.
  - Illumination: Captured between 07:30 and 10:30 local solar time to minimize harsh midday shadows.

---

## 3. Environmental & Microclimate Telemetry

Along with photographic imagery, field scouting teams should log microclimate variables via wireless orchard dataloggers:
- Mean 7-day temperature ($^\circ\text{C}$)
- Mean 7-day relative humidity ($\%$)
- Cumulative 7-day precipitation ($\text{mm}$)
- Canopy wetness duration ($\text{hours/day}$)
These covariates provide the physiological drivers that govern fungal spore germination and lesion expansion velocity.
