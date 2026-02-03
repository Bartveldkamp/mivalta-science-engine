# Operational Tables

This card contains GATC operational tables that support engine operations
but are not part of the 5-axis policy model.

These tables were extracted from legacy v4/v1 cards during the canonical migration.
They use GATC format to preserve exact table names for backward compatibility.

---

## composer_v1 Tables

<!-- GATC:composer_v1:intent_downgrade_path -->
| intent | down_1 | down_2 | down_3 | fallback |
|--------|--------|--------|--------|----------|
| Z8 | Z6 | Z5 | Z4 | REST |
| Z7 | Z6 | Z5 | Z4 | REST |
| Z6 | Z5 | Z4 | Z3 | REST |
| Z5 | Z4 | Z3 | Z2 | REST |
| Z4 | Z3 | Z2 | Z1 | REST |
| Z3 | Z2 | Z1 | R | REST |
| Z2 | Z1 | R | OFF | REST |
| Z1 | R | OFF | OFF | REST |
| R | OFF | OFF | OFF | OFF |
| OFF | OFF | OFF | OFF | OFF |
| REST | OFF | OFF | OFF | OFF |
<!-- /GATC -->

<!-- Source: composer_v1.md -->

<!-- GATC:composer_v1:meso_load_waves -->
| meso_length | slot_pct_min | slot_pct_max | wave_multiplier | wave_name |
|-------------|--------------|--------------|-----------------|-----------|
| 21 | 0 | 25 | 0.90 | ramp_in |
| 21 | 25 | 50 | 1.00 | build |
| 21 | 50 | 75 | 1.08 | peak |
| 21 | 75 | 100 | 0.55 | unload |
| 28 | 0 | 25 | 0.90 | ramp_in |
| 28 | 25 | 50 | 1.00 | build |
| 28 | 50 | 75 | 1.08 | peak |
| 28 | 75 | 100 | 0.55 | unload |
| 14 | 0 | 50 | 0.90 | ramp_in |
| 14 | 50 | 100 | 0.55 | unload |
<!-- /GATC -->

<!-- Source: composer_v1.md -->

<!-- GATC:composer_v1:meso_session_pattern -->
| phase | level_min | level_max | pattern_idx | intent | day_type | notes |
|-------|-----------|-----------|-------------|--------|----------|-------|
| base | 1 | 2 | 0 | Z1 | easy | beginner_base |
| base | 1 | 2 | 1 | OFF | off | beginner_base |
| base | 1 | 2 | 2 | Z2 | easy | beginner_base |
| base | 1 | 2 | 3 | Z1 | easy | beginner_base |
| base | 1 | 2 | 4 | OFF | off | beginner_base |
| base | 1 | 2 | 5 | Z2 | easy | beginner_base |
| base | 1 | 2 | 6 | Z1 | easy | beginner_base |
| base | 3 | 3 | 0 | Z1 | easy | intermediate_base |
| base | 3 | 3 | 1 | Z3 | development | intermediate_base |
| base | 3 | 3 | 2 | Z1 | easy | intermediate_base |
| base | 3 | 3 | 3 | Z2 | easy | intermediate_base |
| base | 3 | 3 | 4 | Z2 | easy | intermediate_base |
| base | 3 | 3 | 5 | Z1 | easy | intermediate_base |
| base | 3 | 3 | 6 | OFF | off | intermediate_base |
| base | 4 | 5 | 0 | Z2 | easy | advanced_base |
| base | 4 | 5 | 1 | Z4 | intensity | advanced_base |
| base | 4 | 5 | 2 | Z1 | easy | advanced_base |
| base | 4 | 5 | 3 | Z2 | easy | advanced_base |
| base | 4 | 5 | 4 | Z2 | easy | advanced_base |
| base | 4 | 5 | 5 | Z1 | easy | advanced_base |
| base | 4 | 5 | 6 | OFF | off | advanced_base |
| build | 1 | 2 | 0 | Z1 | easy | beginner_build |
| build | 1 | 2 | 1 | Z3 | development | beginner_build |
| build | 1 | 2 | 2 | Z1 | easy | beginner_build |
| build | 1 | 2 | 3 | Z2 | easy | beginner_build |
| build | 1 | 2 | 4 | OFF | off | beginner_build |
| build | 1 | 2 | 5 | Z3 | development | beginner_build |
| build | 1 | 2 | 6 | Z1 | easy | beginner_build |
| build | 3 | 3 | 0 | Z1 | easy | intermediate_build |
| build | 3 | 3 | 1 | Z4 | intensity | intermediate_build |
| build | 3 | 3 | 2 | Z2 | easy | intermediate_build |
| build | 3 | 3 | 3 | Z5 | intensity | intermediate_build |
| build | 3 | 3 | 4 | Z1 | easy | intermediate_build |
| build | 3 | 3 | 5 | Z3 | development | intermediate_build |
| build | 3 | 3 | 6 | OFF | off | intermediate_build |
| build | 4 | 5 | 0 | Z2 | easy | advanced_build |
| build | 4 | 5 | 1 | Z5 | intensity | advanced_build |
| build | 4 | 5 | 2 | Z1 | easy | advanced_build |
| build | 4 | 5 | 3 | Z4 | intensity | advanced_build |
| build | 4 | 5 | 4 | Z2 | easy | advanced_build |
| build | 4 | 5 | 5 | Z3 | development | advanced_build |
| build | 4 | 5 | 6 | OFF | off | advanced_build |
| peak | 3 | 5 | 0 | Z1 | easy | peak |
| peak | 3 | 5 | 1 | Z5 | intensity | peak |
| peak | 3 | 5 | 2 | Z1 | easy | peak |
| peak | 3 | 5 | 3 | Z4 | intensity | peak |
| peak | 3 | 5 | 4 | Z2 | easy | peak |
| peak | 3 | 5 | 5 | Z6 | intensity | peak |
| peak | 3 | 5 | 6 | OFF | off | peak |
| taper | 1 | 5 | 0 | Z1 | easy | taper |
| taper | 1 | 5 | 1 | Z4 | intensity | taper |
| taper | 1 | 5 | 2 | OFF | off | taper |
| taper | 1 | 5 | 3 | Z2 | easy | taper |
| taper | 1 | 5 | 4 | OFF | off | taper |
| taper | 1 | 5 | 5 | Z3 | development | taper |
| taper | 1 | 5 | 6 | OFF | off | taper |
| recovery | 1 | 5 | 0 | Z1 | easy | recovery |
| recovery | 1 | 5 | 1 | R | recovery | recovery |
| recovery | 1 | 5 | 2 | Z1 | easy | recovery |
| recovery | 1 | 5 | 3 | OFF | off | recovery |
| recovery | 1 | 5 | 4 | Z2 | easy | recovery |
| recovery | 1 | 5 | 5 | R | recovery | recovery |
| recovery | 1 | 5 | 6 | OFF | off | recovery |
<!-- /GATC -->

<!-- Source: composer_v1.md -->

<!-- GATC:composer_v1:meso_start_rules -->
| phase | level_min | level_max | slot_idx | intent | reason |
|-------|-----------|-----------|----------|--------|--------|
| base | 1 | 5 | 0 | Z1 | ramp_in |
| base | 1 | 5 | 1 | Z2 | ramp_in |
| base | 1 | 5 | 2 | Z1 | ramp_in |
| build | 1 | 5 | 0 | Z2 | prep |
| build | 1 | 5 | 1 | Z1 | prep |
| peak | 1 | 5 | 0 | Z1 | sharpening |
| peak | 1 | 5 | 1 | Z2 | sharpening |
| taper | 1 | 5 | 0 | Z2 | volume_drop |
| taper | 1 | 5 | 1 | R | volume_drop |
| recovery | 1 | 5 | 0 | R | restoration |
| recovery | 1 | 5 | 1 | Z1 | restoration |
| recovery | 1 | 5 | 2 | R | restoration |
<!-- /GATC -->

<!-- Source: composer_v1.md -->

## energy_zones_v4 Tables

<!-- GATC:energy_zones_v4:phase_structure_preferences -->
| phase | preferred_frag_min | preferred_frag_max | preferred_density_min | preferred_density_max | continuous_bias |
|-------|-------------------|-------------------|----------------------|----------------------|-----------------|
| base | 0 | 2 | 0.85 | 1.0 | 0.7 |
| build | 1 | 4 | 0.70 | 0.90 | 0.3 |
| peak | 2 | 6 | 0.60 | 0.85 | 0.1 |
| taper | 0 | 1 | 0.90 | 1.0 | 0.8 |
| recovery | 0 | 0 | 1.0 | 1.0 | 1.0 |
<!-- /GATC -->

<!-- Source: energy_zones_v4.md -->

<!-- GATC:energy_zones_v4:session_prep_multipliers -->
| level | age_min | age_max | warmup_mult | cooldown_mult |
|-------|---------|---------|-------------|---------------|
| beginner | 0 | 39 | 1.2 | 1.1 |
| beginner | 40 | 54 | 1.3 | 1.1 |
| beginner | 55 | 999 | 1.4 | 1.15 |
| novice | 0 | 39 | 1.1 | 1.0 |
| novice | 40 | 54 | 1.2 | 1.05 |
| novice | 55 | 999 | 1.3 | 1.1 |
| intermediate | 0 | 39 | 1.0 | 1.0 |
| intermediate | 40 | 54 | 1.1 | 1.0 |
| intermediate | 55 | 999 | 1.15 | 1.05 |
| advanced | 0 | 39 | 1.0 | 1.0 |
| advanced | 40 | 54 | 1.05 | 1.0 |
| advanced | 55 | 999 | 1.1 | 1.0 |
| elite | 0 | 39 | 1.0 | 1.0 |
| elite | 40 | 54 | 1.0 | 1.0 |
| elite | 55 | 999 | 1.05 | 1.0 |
<!-- /GATC -->

<!-- Source: energy_zones_v4.md -->

<!-- GATC:energy_zones_v4:talk_test -->
| talk_test | zone |
|-----------|------|
| can_sing | R |
| paragraphs_easy | Z1 |
| long_sentences | Z2 |
| short_sentences | Z3 |
| few_words | Z4 |
| cannot_speak | Z5 |
| gasping | Z6 |
| impossible | Z7 |
| max_effort | Z8 |
<!-- /GATC -->

<!-- Source: energy_zones_v4.md -->

<!-- GATC:energy_zones_v4:warmup_cooldown_rules -->
| day_type | base_warmup | max_warmup | base_cooldown | max_cooldown |
|---|---:|---:|---:|---:|
| intensity | 15 | 20 | 10 | 15 |
| easy     | 0  | 0  | 0  | 0  |
| recovery | 0  | 0  | 0  | 0  |
<!-- /GATC -->

<!-- Source: energy_zones_v4.md -->

<!-- GATC:energy_zones_v4:zone_breathing -->
| zone | breathing_cue | breaths_per_min_min | breaths_per_min_max |
|------|---------------|---------------------|---------------------|
| R | Natural, effortless breathing. Could hold a conversation or hum a tune. | 10 | 20 |
| Z1 | Relaxed, rhythmic breathing. Nose breathing possible. | 15 | 25 |
| Z2 | Controlled breathing, slightly deeper. Nose or mouth. | 20 | 30 |
| Z3 | Deliberate breathing, mouth preferred. Notice the effort. | 30 | 40 |
| Z4 | Deep, rhythmic mouth breathing. Controlled but working. | 40 | 50 |
| Z5 | Heavy, rapid breathing. Cannot maintain a conversation. | 50 | 60 |
| Z6 | Gasping for air. Maximum ventilation. Recovery breathing between reps. | 50 | 60 |
| Z7 | Maximum ventilation. Brief maximal efforts. Full recovery between reps. | 50 | 60 |
| Z8 | Breath-hold during effort, explosive recovery. Pure speed. | 50 | 60 |
<!-- /GATC -->

<!-- Source: energy_zones_v4.md -->

<!-- GATC:energy_zones_v4:zone_feel -->
| zone | feel_short | feel_detailed | cue_start | cue_during | cue_end |
|------|------------|---------------|-----------|------------|---------|
| R | Easy movement | Barely any effort. Could do this all day. Recovery only. | Just move gently | Stay completely relaxed | Good recovery work |
| Z1 | Very easy | Comfortable conversation possible. Building aerobic base. | Find your easy rhythm | Full sentences OK | Steady and relaxed |
| Z2 | Comfortable | Slight effort but sustainable for hours. Long sentences possible. | Settle into your pace | Controlled breathing | Maintain this feeling |
| Z3 | Working | Purposeful effort. Short sentences only. Tempo feeling. | Build into the effort | Keep control, short phrases | Ease off smoothly |
| Z4 | Hard | Few words only. Controlled discomfort. Threshold effort. | Build over first 30 seconds | Hold steady, stay focused | You earned that rest |
| Z5 | Very hard | Cannot talk. Racing heart. VO2max effort. | Attack from the start | Maximum sustainable effort | Recover fully |
| Z6 | Gasping | Near-maximum power. Short bursts only. Anaerobic burn. | Explode into it | Give everything | Complete rest needed |
| Z7 | Maximum | Peak power output. Seconds only. All-out effort. | Full power immediately | Absolute maximum | Stop and recover |
| Z8 | Sprint | Absolute max speed. 5-15 seconds only. Neuromuscular. | Maximum acceleration | Pure speed | Done, walk it off |
<!-- /GATC -->

<!-- Source: energy_zones_v4.md -->

<!-- GATC:energy_zones_v4:zone_interval_constraints -->
| zone | work_min_min | work_min_max | rest_min_min | rest_min_max | rest_ratio_min | rest_ratio_max | is_continuous | can_be_intervals |
|------|-------------|-------------|-------------|-------------|---------------|---------------|--------------|-----------------|
| R | 15 | 30 | 0 | 0 | 0.0 | 0.0 | true | false |
| Z1 | 10 | 180 | 1 | 4 | 0.0 | 0.2 | true | true |
| Z2 | 10 | 180 | 1 | 4 | 0.0 | 0.2 | true | true |
| Z3 | 6 | 30 | 2 | 8 | 0.25 | 1.0 | false | true |
| Z4 | 4 | 20 | 2 | 10 | 0.5 | 1.5 | false | true |
| Z5 | 1 | 5 | 1 | 6 | 1.0 | 4.0 | false | true |
| Z6 | 0.5 | 1.5 | 2 | 8 | 2.0 | 8.0 | false | true |
| Z7 | 0.17 | 0.5 | 2 | 6 | 6.0 | 20.0 | false | true |
| Z8 | 0.05 | 0.17 | 2 | 10 | 20.0 | 60.0 | false | true |
<!-- /GATC -->

<!-- Source: energy_zones_v4.md -->

## goal_specificity_v4 Tables

<!-- GATC:goal_specificity_v4:goal_limiter_mapping -->
| sport | goal_type | duration_min | duration_max | primary_limiters | secondary_limiters | protected_intents | prefer_intents | allowed_intents | deprioritized_intents | never_intents | notes |
|-------|-----------|--------------|--------------|------------------|--------------------|--------------------|----------------|-----------------|----------------------|---------------|-------|
| running | 1500m | 3 | 6 | vo2max,neuromuscular | anaerobic_capacity | Z5,Z6,Z8 | Z4,Z7 | Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8,R | Z2 | | high_speed |
| running | 5k | 12 | 30 | vo2max,threshold | economy | Z5,Z4 | Z3,Z8 | Z1,Z2,Z3,Z4,Z5,Z8,R | Z2 | Z6,Z7 | speed_endurance |
| running | 10k | 30 | 55 | threshold,vo2max | aerobic_base | Z4,Z5 | Z3,Z2 | Z1,Z2,Z3,Z4,Z5,R | Z8 | Z6,Z7 | threshold_dominant |
| running | half_marathon | 55 | 120 | threshold,aerobic_base | fatigue_resistance | Z3,Z4 | Z2 | Z1,Z2,Z3,Z4,R | Z5 | Z6,Z7,Z8 | tempo_focus |
| running | marathon | 120 | 300 | aerobic_base,fatigue_resistance | threshold | Z2,Z3 | Z1,Z4 | Z1,Z2,Z3,Z4,R | Z5 | Z6,Z7,Z8 | aerobic_priority |
| running | ultra | 300 | 2000 | fatigue_resistance,aerobic_base | durability | Z2,R | Z1,Z3 | Z1,Z2,Z3,R | Z4 | Z5,Z6,Z7,Z8 | durability_focus |
| cycling | crit | 30 | 60 | anaerobic,vo2max | neuromuscular | Z5,Z6 | Z4,Z7 | Z1,Z2,Z3,Z4,Z5,Z6,Z7,R | | Z8 | repeatability |
| cycling | road_race | 120 | 300 | threshold,aerobic_base | vo2max | Z3,Z4 | Z2,Z5 | Z1,Z2,Z3,Z4,Z5,R | Z6 | Z7,Z8 | sustained_power |
| cycling | tt | 15 | 60 | threshold | vo2max | Z4 | Z3,Z5 | Z1,Z2,Z3,Z4,Z5,R | | Z6,Z7,Z8 | steady_state |
| cycling | gran_fondo | 180 | 480 | aerobic_base,fatigue_resistance | threshold | Z2,Z3 | Z1 | Z1,Z2,Z3,Z4,R | Z5 | Z6,Z7,Z8 | long_endurance |
<!-- /GATC -->

<!-- Source: goal_specificity_v4.md -->

## interval_generator_v1 Tables

<!-- GATC:interval_generator_v1:guardrails -->
| guardrail_id | condition | blocked_action | severity | reason |
|--------------|-----------|----------------|----------|--------|
| no_z5_continuous | zone in Z5,Z6,Z7,Z8 | continuous=true | critical | Physiologically invalid |
| beginner_intensity | level=beginner | zone in Z5,Z6,Z7,Z8 | critical | Progression required |
| novice_anaerobic | level=novice | zone in Z6,Z7 | critical | Insufficient base |
| beginner_patterns | level=beginner | pattern in wave,cluster,ladder,pyramid | high | Mental overload |
| novice_density | level=novice | pattern in wave,cluster | high | Complexity limit |
| z7_rest_floor | zone=Z7 | rest_ratio < 3.0 | critical | Insufficient recovery |
| z8_rest_floor | zone=Z8 | rest_ratio < 6.0 | critical | Insufficient recovery |
| z6_work_ceiling | zone=Z6 | work_sec > 90 | critical | Exceeds anaerobic capacity |
| z7_work_ceiling | zone=Z7 | work_sec > 30 | critical | Exceeds neuromuscular capacity |
| z8_work_ceiling | zone=Z8 | work_sec > 15 | critical | Exceeds sprint capacity |
| red_readiness | readiness=red | zone in Z4,Z5,Z6,Z7,Z8 | critical | Recovery required |
| amber_density | readiness=amber | pattern in wave,cluster | high | Reduce stress |
| max_tiz_exceeded | tiz > zone.max_tiz_min | generate | critical | Overload prevention |
| micro_hiit_zone | session_class=micro_hiit | zone in Z3,Z4 | high | Wrong class for zone |
<!-- /GATC -->

<!-- Source: interval_generator_v1.md -->

<!-- GATC:interval_generator_v1:meso_position_modifiers -->
| position_band | pattern_filter | rest_ratio_mod | complexity_cap | notes |
|---------------|----------------|----------------|----------------|-------|
| early | equal,descending | 1.1 | low | Establish load safely |
| mid | all | 1.0 | none | Full variety |
| late | equal,descending | 1.2 | medium | Pre-unload simplicity |
| unload | equal | 1.3 | low | Recovery priority |
<!-- /GATC -->

<!-- Source: interval_generator_v1.md -->

<!-- GATC:interval_generator_v1:patterns -->
| pattern_id | formula_type | description | mental_load | level_min | phases_allowed |
|------------|--------------|-------------|-------------|-----------|----------------|
| equal | constant | All intervals same duration: [w] × n | low | beginner | all |
| descending | linear_down | Duration decreases: [w, w-Δ, w-2Δ, ...] | low | beginner | all |
| ladder | linear_up | Duration increases: [w, w+Δ, w+2Δ, ...] | medium | novice | build,peak |
| pyramid | mirror | Up then down: [w, w+Δ, ..., w+Δ, w] | medium | novice | build,peak |
| wave | repeating | Motif repeats: [w1,w2,w3] × m | high | intermediate | build,peak |
| cluster | nested | Sets of micro: R × (r × w/rest) | high | intermediate | peak |
| fartlek | variable | Varied durations within bounds: [w1,w2,w3,...] | medium | beginner | all |
| progressive | increasing | Each rep harder/longer: [w, w×1.1, w×1.2, ...] | high | advanced | peak |
<!-- /GATC -->

<!-- Source: interval_generator_v1.md -->

<!-- GATC:interval_generator_v1:primitives -->
| primitive_id | work_sec_min | work_sec_max | step_sec | primary_zones | level_min | session_classes |
|--------------|--------------|--------------|----------|---------------|-----------|-----------------|
| micro | 5 | 60 | 5 | Z5,Z6,Z7,Z8 | novice | micro_hiit,micro_quality,standard |
| short | 60 | 240 | 15 | Z4,Z5,Z6 | novice | micro_quality,commute_tempo,standard |
| medium | 240 | 600 | 30 | Z3,Z4,Z5 | beginner | commute_tempo,commute_easy,standard |
| long | 600 | 1200 | 60 | Z3,Z4 | beginner | commute_easy,standard |
<!-- /GATC -->

<!-- Source: interval_generator_v1.md -->

<!-- GATC:interval_generator_v1:research_protocols -->
| protocol_id | zone | pattern | work_sec | rest_ratio | reps | sets | source | level_min | phases |
|-------------|------|---------|----------|------------|------|------|--------|-----------|--------|
| billat_30_30 | Z5 | equal | 30 | 1.0 | 20 | 1 | Billat 2000 | intermediate | build,peak |
| ronnestad_30_15 | Z5 | equal | 30 | 0.5 | 13 | 3 | Rønnestad 2014 | advanced | peak |
| tabata_20_10 | Z6 | equal | 20 | 0.5 | 8 | 1 | Tabata 1996 | elite | peak |
| norwegian_4x16 | Z4 | equal | 960 | 0.25 | 4 | 1 | Norwegian Model | advanced | build,peak |
| classic_4x4 | Z5 | equal | 240 | 0.75 | 4 | 1 | Consensus | novice | all |
| classic_5x3 | Z5 | equal | 180 | 1.0 | 5 | 1 | Consensus | novice | all |
| beginner_90_90 | Z5 | equal | 90 | 1.0 | 6 | 1 | Beginner intro | novice | build |
| beginner_60_60 | Z4 | equal | 60 | 1.0 | 8 | 1 | Beginner intro | beginner | build |
| pyramid_1_2_3 | Z5 | pyramid | 60 | 1.0 | 5 | 1 | Coach standard | novice | build,peak |
| descending_4_3_2_1 | Z5 | descending | 240 | 0.75 | 4 | 1 | Mental ease | novice | all |
<!-- /GATC -->

<!-- Source: interval_generator_v1.md -->

<!-- GATC:interval_generator_v1:selection_policy -->
| phase | level | primitives_allowed | patterns_allowed | rest_goal | density_bias |
|-------|-------|-------------------|------------------|-----------|--------------|
| base | beginner | long | equal | quality | 0.0 |
| base | novice | medium,long | equal | quality | 0.0 |
| base | intermediate | medium,long | equal,descending | quality | 0.1 |
| base | advanced | medium,long | equal,descending | quality | 0.2 |
| base | elite | medium,long | equal,descending,ladder | balanced | 0.3 |
| build | beginner | medium | equal | quality | 0.1 |
| build | novice | short,medium | equal,descending | balanced | 0.2 |
| build | intermediate | short,medium | equal,descending,ladder,pyramid | balanced | 0.3 |
| build | advanced | micro,short,medium | equal,ladder,pyramid,wave | balanced | 0.4 |
| build | elite | micro,short,medium | all | density | 0.5 |
| peak | beginner | medium | equal | quality | 0.2 |
| peak | novice | short,medium | equal,descending | balanced | 0.3 |
| peak | intermediate | micro,short | equal,pyramid,wave | density | 0.5 |
| peak | advanced | micro,short | equal,pyramid,wave,cluster | density | 0.6 |
| peak | elite | micro,short | all | density | 0.7 |
| unload | beginner | long | equal | quality | 0.0 |
| unload | novice | medium,long | equal | quality | 0.0 |
| unload | intermediate | medium,long | equal | quality | 0.1 |
| unload | advanced | medium,long | equal,descending | quality | 0.1 |
| unload | elite | medium,long | equal,descending | quality | 0.2 |
<!-- /GATC -->

<!-- Source: interval_generator_v1.md -->

<!-- GATC:interval_generator_v1:session_class_modifiers -->
| session_class | primitives_override | patterns_override | max_work_sec | rest_ratio_mod | notes |
|---------------|--------------------|--------------------|--------------|----------------|-------|
| standard | none | none | none | 1.0 | Full range available |
| micro_hiit | micro | equal,wave | 60 | 0.8 | Short, dense intervals |
| micro_quality | micro,short | equal | 120 | 1.0 | Quality micro work |
| commute_easy | medium,long | equal | none | 1.2 | Longer rest, simpler |
| commute_tempo | short,medium | equal,descending | 600 | 1.0 | Moderate complexity |
| vilpa | micro | equal | 30 | 0.5 | Ultra-short bursts |
| exercise_snack | micro | equal | 60 | 1.0 | Brief quality efforts |
<!-- /GATC -->

<!-- Source: interval_generator_v1.md -->

<!-- GATC:interval_generator_v1:zone_constraints -->
| zone | work_sec_min | work_sec_max | rest_ratio_min | rest_ratio_max | max_tiz_min | max_reps | continuous_allowed | continuous_max_min | primitives_allowed |
|------|--------------|--------------|----------------|----------------|-------------|----------|--------------------|--------------------|-------------------|
| R | 900 | 2700 | 0.0 | 0.0 | 45 | 1 | true | 45 | long |
| Z1 | 600 | 10800 | 0.0 | 0.2 | 180 | 4 | true | 180 | long,medium |
| Z2 | 600 | 7200 | 0.0 | 0.2 | 120 | 4 | true | 120 | long,medium |
| Z3 | 360 | 1800 | 0.17 | 0.33 | 45 | 4 | true | 45 | medium,long |
| Z4 | 240 | 1200 | 0.25 | 0.50 | 60 | 6 | true | 30 | short,medium,long |
| Z5 | 60 | 300 | 0.75 | 1.25 | 25 | 12 | false | 0 | micro,short |
| Z6 | 30 | 90 | 2.0 | 3.0 | 12 | 15 | false | 0 | micro |
| Z7 | 10 | 30 | 3.0 | 6.0 | 5 | 20 | false | 0 | micro |
| Z8 | 5 | 15 | 6.0 | 18.0 | 2 | 30 | false | 0 | micro |
<!-- /GATC -->

<!-- Source: interval_generator_v1.md -->

## micro_training_commute_v4 Tables

<!-- GATC:micro_training_commute_v4:micro_med -->
| zone | standard_med_min | micro_med_min | micro_max_per_session | notes |
|------|------------------|---------------|-----------------------|-------|
| Z1 | 20 | 10 | 30 | easy_aerobic |
| Z2 | 20 | 10 | 30 | aerobic_base |
| Z3 | 10 | 5 | 15 | tempo_threshold |
| Z4 | 8 | 4 | 12 | threshold_intervals |
| Z5 | 4 | 2 | 8 | vo2max_bursts |
| Z6 | 2 | 1 | 4 | anaerobic |
| Z8 | 0.5 | 0.2 | 2 | neuromuscular_sprints |
<!-- /GATC -->

<!-- Source: micro_training_commute_v4.md -->

<!-- GATC:micro_training_commute_v4:readiness_gates -->
| readiness | micro_hiit_allowed | z5_z8_allowed | recommended_class | notes |
|-----------|--------------------| --------------|-------------------|-------|
| green | true | true | any | full_options |
| amber | true | false | commute_easy | z1_z2_only |
| red | false | false | rest_or_walk | no_cycling_intensity |
<!-- /GATC -->

<!-- Source: micro_training_commute_v4.md -->

<!-- GATC:micro_training_commute_v4:spacing_rules -->
| session_class | min_hours_between_same | min_hours_before_key | min_hours_after_key | notes |
|---------------|------------------------|----------------------|---------------------|-------|
| commute_easy | 0 | 0 | 0 | no_restrictions |
| commute_tempo | 12 | 24 | 12 | light_recovery_needed |
| micro_quality | 48 | 48 | 24 | structured_threshold_work |
| micro_hiit | 48 | 48 | 24 | full_intensity_spacing |
| exercise_snack | 4 | 12 | 4 | minimal_impact |
| vilpa | 0 | 0 | 0 | incidental_no_planning |
<!-- /GATC -->

<!-- Source: micro_training_commute_v4.md -->

<!-- GATC:micro_training_commute_v4:zone_eligibility -->
| session_class | r_eligible | z1_eligible | z2_eligible | z3_eligible | z4_eligible | z5_eligible | z6_eligible | z7_eligible | z8_eligible |
|---------------|------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| commute_easy | true | true | true | false | false | false | false | false | false |
| commute_tempo | true | true | true | true | false | false | false | false | false |
| micro_quality | true | true | true | true | true | true | false | false | false |
| micro_hiit | false | false | true | false | true | true | true | false | true |
| exercise_snack | false | false | true | false | false | true | false | false | true |
| vilpa | false | true | true | false | false | false | false | false | true |
<!-- /GATC -->

<!-- Source: micro_training_commute_v4.md -->

## ntiz_dose_v4 Tables

<!-- GATC:ntiz_dose_v4:beginner_progression_meso -->
| meso_num | r_z2_pct_min | z3_allowed | z3_max_per_session | z4_allowed | z4_max_per_meso | z5_z8_allowed | rationale |
|----------|--------------|------------|--------------------| -----------|-----------------|---------------|-----------|
| 1 | 100 | false | 0 | false | 0 | false | habit_formation |
| 2 | 95 | true | 6 | false | 0 | false | introduce_tempo |
| 3 | 90 | true | 6 | true | 20 | false | introduce_threshold |
| 4 | 85 | true | 12 | true | 36 | false | graduated_intensity |
| 5 | 85 | true | 12 | true | 36 | false | continued_progression |
| 6 | 80 | true | -1 | true | -1 | true | graduate_to_novice |
<!-- /GATC -->

<!-- Source: ntiz_dose_v4.md -->

## ntiz_master_v4 Tables

<!-- GATC:ntiz_master_v4:zone_credit -->
| zone | ntiz_credit | category | purpose |
|------|-------------|----------|---------|
| R | false | restoration | Active restoration, NOT training |
| Z1 | true | aerobic | Easy aerobic, peripheral adaptations |
| Z2 | true | aerobic | Aerobic development, technique |
| Z3 | true | tempo | Tempo, lactate clearance |
| Z4 | true | threshold | Threshold intervals |
| Z5 | true | vo2max | VO2max intervals |
| Z6 | true | anaerobic | Anaerobic capacity |
| Z7 | true | anaerobic | Max anaerobic power |
| Z8 | true | neuromuscular | Neuromuscular, speed |
<!-- /GATC -->

<!-- Source: ntiz_master_v4.md -->

## pattern_policy_v1 Tables

<!-- GATC:pattern_policy_v1:dose_envelopes -->
| sport | zone | level | phase | ntiz_success_min | ntiz_success_max | rest_ratio_bias | max_reps_preferred | min_work_sec | max_work_sec | density_bias | source | updated |
|-------|------|-------|-------|------------------|------------------|-----------------|-------------------|--------------|--------------|--------------|--------|---------|
| running | Z5 | beginner | base | 8 | 12 | 1.0 | 6 | 60 | 180 | 0.4 | default | 2025-01-01 |
| running | Z5 | beginner | build | 10 | 14 | 1.0 | 8 | 60 | 180 | 0.45 | default | 2025-01-01 |
| running | Z5 | novice | base | 10 | 15 | 1.0 | 8 | 60 | 240 | 0.45 | default | 2025-01-01 |
| running | Z5 | novice | build | 12 | 18 | 0.9 | 10 | 60 | 240 | 0.5 | default | 2025-01-01 |
| running | Z5 | intermediate | base | 12 | 18 | 1.0 | 8 | 90 | 300 | 0.5 | default | 2025-01-01 |
| running | Z5 | intermediate | build | 14 | 20 | 0.9 | 10 | 90 | 300 | 0.55 | default | 2025-01-01 |
| running | Z5 | intermediate | peak | 12 | 16 | 1.0 | 8 | 120 | 300 | 0.5 | default | 2025-01-01 |
| running | Z5 | advanced | base | 14 | 20 | 1.0 | 10 | 90 | 300 | 0.55 | default | 2025-01-01 |
| running | Z5 | advanced | build | 16 | 24 | 0.85 | 12 | 90 | 300 | 0.6 | default | 2025-01-01 |
| running | Z5 | advanced | peak | 14 | 18 | 1.0 | 10 | 120 | 300 | 0.5 | default | 2025-01-01 |
| running | Z4 | beginner | base | 10 | 15 | 1.0 | 6 | 120 | 300 | 0.4 | default | 2025-01-01 |
| running | Z4 | intermediate | base | 15 | 25 | 0.8 | 8 | 180 | 480 | 0.5 | default | 2025-01-01 |
| running | Z4 | intermediate | build | 18 | 30 | 0.75 | 10 | 180 | 480 | 0.55 | default | 2025-01-01 |
| running | Z6 | intermediate | build | 4 | 8 | 1.5 | 8 | 30 | 90 | 0.35 | default | 2025-01-01 |
| running | Z6 | advanced | build | 6 | 10 | 1.25 | 10 | 30 | 120 | 0.4 | default | 2025-01-01 |
| cycling | Z5 | intermediate | build | 14 | 20 | 1.0 | 8 | 120 | 300 | 0.5 | default | 2025-01-01 |
| cycling | Z4 | intermediate | base | 20 | 35 | 0.75 | 8 | 240 | 600 | 0.5 | default | 2025-01-01 |
<!-- /GATC -->

<!-- Source: pattern_policy_v1.md -->

<!-- GATC:pattern_policy_v1:pattern_gates -->
| sport | zone | level | readiness | forbidden_patterns | rest_ratio_min_override | rest_ratio_max_override | notes |
|-------|------|-------|-----------|-------------------|-------------------------|-------------------------|-------|
| all | Z5 | all | amber | cluster,wave,fartlek | 1.0 | 1.5 | Reduce complexity when fatigued |
| all | Z5 | all | red | ALL | | | No Z5 when red |
| all | Z6 | all | amber | cluster,wave,progressive | 1.25 | 2.0 | Extra rest when fatigued |
| all | Z6 | all | red | ALL | | | No Z6 when red |
| all | Z7 | all | amber | ALL | | | No Z7 when amber |
| all | Z7 | all | red | ALL | | | No Z7 when red |
| all | Z4 | beginner | amber | ladder,pyramid,wave | 1.0 | 1.25 | Simplify for beginners |
| all | Z4 | beginner | red | cluster,wave,ladder,pyramid | 1.0 | 1.5 | Very simple only |
| all | Z5 | beginner | green | cluster,wave,progressive | | | Too complex for beginners |
| all | Z5 | novice | green | cluster,progressive | | | Limit complexity |
<!-- /GATC -->

<!-- Source: pattern_policy_v1.md -->

<!-- GATC:pattern_policy_v1:pattern_weights -->
| sport | zone | level | phase | session_class | equal | descending | ladder | pyramid | wave | cluster | fartlek | progressive | confidence | updated |
|-------|------|-------|-------|---------------|-------|------------|--------|---------|------|---------|---------|-------------|------------|---------|
| running | Z5 | beginner | base | standard | 0.70 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 | 0.00 | 0.5 | 2025-01-01 |
| running | Z5 | beginner | build | standard | 0.60 | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.15 | 0.00 | 0.5 | 2025-01-01 |
| running | Z5 | novice | base | standard | 0.50 | 0.20 | 0.15 | 0.10 | 0.00 | 0.00 | 0.05 | 0.00 | 0.5 | 2025-01-01 |
| running | Z5 | novice | build | standard | 0.40 | 0.20 | 0.20 | 0.15 | 0.00 | 0.00 | 0.05 | 0.00 | 0.5 | 2025-01-01 |
| running | Z5 | intermediate | base | standard | 0.35 | 0.20 | 0.15 | 0.15 | 0.10 | 0.00 | 0.05 | 0.00 | 0.6 | 2025-01-01 |
| running | Z5 | intermediate | build | standard | 0.30 | 0.15 | 0.15 | 0.15 | 0.15 | 0.05 | 0.05 | 0.00 | 0.6 | 2025-01-01 |
| running | Z5 | intermediate | peak | standard | 0.25 | 0.15 | 0.15 | 0.15 | 0.15 | 0.10 | 0.05 | 0.00 | 0.6 | 2025-01-01 |
| running | Z5 | advanced | base | standard | 0.25 | 0.15 | 0.15 | 0.15 | 0.15 | 0.10 | 0.05 | 0.00 | 0.7 | 2025-01-01 |
| running | Z5 | advanced | build | standard | 0.20 | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 | 0.05 | 0.00 | 0.7 | 2025-01-01 |
| running | Z5 | advanced | peak | standard | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 | 0.05 | 0.05 | 0.7 | 2025-01-01 |
| running | Z4 | beginner | base | standard | 0.80 | 0.15 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 0.00 | 0.5 | 2025-01-01 |
| running | Z4 | novice | base | standard | 0.60 | 0.20 | 0.10 | 0.05 | 0.00 | 0.00 | 0.05 | 0.00 | 0.5 | 2025-01-01 |
| running | Z4 | intermediate | base | standard | 0.40 | 0.20 | 0.15 | 0.10 | 0.10 | 0.00 | 0.05 | 0.00 | 0.6 | 2025-01-01 |
| running | Z4 | intermediate | build | standard | 0.35 | 0.20 | 0.15 | 0.10 | 0.10 | 0.05 | 0.05 | 0.00 | 0.6 | 2025-01-01 |
| running | Z6 | intermediate | build | standard | 0.40 | 0.20 | 0.15 | 0.10 | 0.10 | 0.05 | 0.00 | 0.00 | 0.6 | 2025-01-01 |
| running | Z6 | advanced | build | standard | 0.30 | 0.20 | 0.15 | 0.15 | 0.10 | 0.10 | 0.00 | 0.00 | 0.7 | 2025-01-01 |
| cycling | Z5 | intermediate | build | standard | 0.35 | 0.20 | 0.15 | 0.15 | 0.10 | 0.05 | 0.00 | 0.00 | 0.6 | 2025-01-01 |
| cycling | Z4 | intermediate | base | standard | 0.45 | 0.20 | 0.15 | 0.10 | 0.10 | 0.00 | 0.00 | 0.00 | 0.6 | 2025-01-01 |
| running | Z5 | intermediate | build | micro_hiit | 0.50 | 0.20 | 0.00 | 0.00 | 0.20 | 0.00 | 0.10 | 0.00 | 0.5 | 2025-01-01 |
| running | Z5 | intermediate | build | vilpa | 0.70 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 0.10 | 0.00 | 0.5 | 2025-01-01 |
<!-- /GATC -->

<!-- Source: pattern_policy_v1.md -->

## planner_policy_v4 Tables

<!-- GATC:planner_policy_v4:default_values -->
| parameter | value | unit | notes |
|-----------|-------|------|-------|
| default_taper_days | 14 | days | Used when event type not found |
| default_meso_days | 28 | days | Standard mesocycle length |
| intensity_cap_rounding | ceil | - | ceil floor or round |
| budget_rounding_decimals | 1 | - | Decimal places for minutes |
| shortfall_warning_pct | 10 | % | Gabbett ACWR safe zone; <10% variation negligible |
<!-- /GATC -->

<!-- Source: planner_policy_v4.md -->

<!-- GATC:planner_policy_v4:paradigm_selector -->
| min_session_max | max_session_max | min_sessions_per_7d | max_sessions_per_7d | paradigm         | max_quality_per_7d | max_micro_pct_meso | allow_long_run |
|-----------------|-----------------|---------------------|---------------------|------------------|--------------------|--------------------|----------------|
| 0               | 60              | 1                   | 14                  | micro_structured | 2                  | 0.40               | false          |
| 75              | 999             | 8                   | 14                  | high_volume      | 4                  | 0.00               | true           |
| 0               | 999             | 1                   | 14                  | standard         | 3                  | 0.00               | true           |
<!-- /GATC -->

<!-- Source: planner_policy_v4.md -->

<!-- GATC:planner_policy_v4:senior_safe_phases -->
| age_min | age_max | level | original_phase | safe_alternative | max_intensity | notes |
|---------|---------|-------|----------------|------------------|---------------|-------|
| 65 | 99 | beginner | build | base_extended | Z2 | no_threshold_work |
| 65 | 99 | beginner | peak | base_extended | Z2 | no_vo2max_work |
| 65 | 99 | beginner | topshape | base_extended | Z2 | maintain_aerobic |
| 65 | 99 | novice | build | base_extended | Z3 | limited_tempo |
| 65 | 99 | novice | peak | base_extended | Z3 | limited_tempo |
| 55 | 64 | beginner | build | build_modified | Z3 | tempo_max_no_threshold |
| 55 | 64 | beginner | peak | build_modified | Z3 | tempo_max_no_threshold |
| 0 | 54 | beginner | build | build | Z4 | standard_beginner_build |
| 0 | 54 | beginner | peak | peak | Z4 | standard_beginner_peak |
<!-- /GATC -->

<!-- Source: planner_policy_v4.md -->

<!-- GATC:planner_policy_v4:session_frequency_caps -->
| level | age_min | age_max | max_sessions_per_7d | min_rest_days_per_7d | notes |
|-------|---------|---------|---------------------|----------------------|-------|
| beginner | 0 | 39 | 4 | 3 | tissue_adaptation_needed |
| beginner | 40 | 54 | 4 | 3 | masters_recovery |
| beginner | 55 | 64 | 3 | 4 | senior_recovery |
| beginner | 65 | 99 | 3 | 4 | veteran_recovery |
| novice | 0 | 39 | 4 | 3 | building_base |
| novice | 40 | 54 | 4 | 3 | masters_recovery |
| novice | 55 | 64 | 3 | 4 | senior_recovery |
| novice | 65 | 99 | 3 | 4 | veteran_recovery |
| intermediate | 0 | 39 | 5 | 2 | established_base |
| intermediate | 40 | 54 | 5 | 2 | masters_intermediate |
| intermediate | 55 | 64 | 4 | 3 | senior_intermediate |
| intermediate | 65 | 99 | 4 | 3 | veteran_intermediate |
| advanced | 0 | 39 | 6 | 1 | high_volume_tolerance |
| advanced | 40 | 54 | 5 | 2 | masters_advanced |
| advanced | 55 | 64 | 5 | 2 | senior_advanced |
| advanced | 65 | 99 | 4 | 3 | veteran_advanced |
| elite | 0 | 39 | 7 | 0 | professional_tolerance |
| elite | 40 | 54 | 6 | 1 | masters_elite |
| elite | 55 | 64 | 5 | 2 | senior_elite |
| elite | 65 | 99 | 5 | 2 | veteran_elite |
| pro | 0 | 39 | 10 | 1 | double_session_support |
| pro | 40 | 54 | 8 | 1 | masters_pro |
<!-- /GATC -->

<!-- Source: planner_policy_v4.md -->

## recovery_v4 Tables

<!-- GATC:recovery_v4:age_d_max_modifiers -->
| age_min | age_max | d_max_mult | notes | source |
|---------|---------|-----------:|-------|--------|
| 0 | 39 | 1.00 | baseline | legacy_baseline |
| 40 | 49 | 0.92 | modest reduction | legacy_baseline |
| 50 | 999 | 0.85 | conservative reduction | legacy_baseline |
<!-- /GATC -->

<!-- Source: recovery_v4.md -->

<!-- GATC:recovery_v4:decay_parameters -->
| parameter | value | time_constant_days |
|-----------|-------|-------------------|
| lambda | 0.024 | 42 |
| mu | 0.143 | 7 |
<!-- /GATC -->

<!-- Source: recovery_v4.md -->

<!-- GATC:recovery_v4:level_decay_modifiers -->
| level | fitness_mult | fatigue_mult | notes | source |
|-------|-------------:|-------------:|-------|--------|
| beginner | 1.00 | 0.85 | slower fatigue dissipation | research_banister |
| novice | 1.00 | 0.90 | moderate fatigue dissipation | research_banister |
| intermediate | 1.00 | 1.00 | baseline | research_banister |
| advanced | 1.00 | 1.10 | faster fatigue dissipation | research_banister |
| elite | 1.00 | 1.15 | optimized recovery | research_banister |
<!-- /GATC -->

<!-- Source: recovery_v4.md -->

<!-- GATC:recovery_v4:readiness_gated_sessions -->
| readiness_level | intent | zone | duration_min | notes |
|-----------------|--------|------|--------------|-------|
| red | rest | REST | 0 | Complete rest, no training |
| orange | recovery | R | 20 | Active recovery only, gentle movement |
| yellow | easy | Z1 | 30 | Easy aerobic, no intensity |
| green | normal | Z2 | 45 | Normal training allowed |
<!-- /GATC -->

<!-- Source: recovery_v4.md -->

<!-- GATC:recovery_v4:readiness_thresholds -->
| metric | green_min | green_max | amber_min | amber_max | red_threshold |
|--------|-----------|-----------|-----------|-----------|---------------|
| hrv_pct_baseline | -10 | 10 | -20 | -10 | -20 |
| rhr_bpm_vs_baseline | -5 | 5 | 5 | 10 | 10 |
| sleep_hours | 7 | 99 | 5 | 7 | 5 |
| sleep_quality_10 | 6 | 10 | 4 | 6 | 4 |
| wellness_100 | 70 | 100 | 50 | 70 | 50 |
| soreness | 0 | 1 | 2 | 3 | 4 |
<!-- /GATC -->

<!-- Source: recovery_v4.md -->

<!-- GATC:recovery_v4:same_zone_intensity -->
| zone | min_hours | preferred_hours | notes | evidence_level |
|------|-----------|-----------------|-------|----------------|
| Z4 | 48 | 72 | threshold tolerates 2x/week if spaced | moderate |
| Z5 | 48 | 72 | VO2max intervals, autonomic recovery 24-48h | moderate |
| Z6 | 72 | 96 | high glycolytic + mechanical stress | moderate |
| Z7 | 72 | 96 | neuromuscular fatigue up to 72h | strong |
| Z8 | 24 | 48 | pure alactic can repeat sooner if low volume | limited |
<!-- /GATC -->

<!-- Source: recovery_v4.md -->

## viterbi_policy Tables

<!-- GATC:viterbi_policy:confidence_thresholds -->
| parameter | threshold | description | source |
|-----------|-----------|-------------|--------|
| state_trust_min | 0.70 | Minimum confidence to trust Viterbi state for gating | warm_up_ux |
| baseline_ready_days | 14 | Days of data needed for confident baseline | foster_1998 |
<!-- /GATC -->

Notes:
- `state_trust_min`: Below this confidence, composer uses default (neutral) gating
- `baseline_ready_days`: Aligns with Foster's 2-week baseline window for load monitoring

<!-- Source: viterbi_policy.md -->

## session_budgets_v4 Tables

<!-- GATC:session_budgets_v4:meso_session_budgets -->
| paradigm | phase | level_min | level_max | age_min | age_max | quality_pct | z3_max_meso | z4_max_meso | z5_max_meso | z6z8_max_meso | min_z2_sessions | min_recovery_gap_days | source |
|----------|-------|-----------|-----------|---------|---------|-------------|-------------|-------------|-------------|---------------|-----------------|----------------------|--------|
| standard | base | 1 | 2 | 0 | 999 | 10 | 2 | 0 | 0 | 0 | 8 | 2 | seiler_2010 |
| standard | base | 3 | 3 | 0 | 39 | 15 | 3 | 1 | 0 | 0 | 6 | 2 | seiler_2010 |
| standard | base | 3 | 3 | 40 | 54 | 12 | 2 | 1 | 0 | 0 | 6 | 2 | friel_masters |
| standard | base | 3 | 3 | 55 | 999 | 10 | 2 | 0 | 0 | 0 | 6 | 3 | friel_masters |
| standard | base | 4 | 5 | 0 | 39 | 15 | 3 | 2 | 0 | 0 | 5 | 2 | seiler_2010 |
| standard | base | 4 | 5 | 40 | 999 | 12 | 2 | 1 | 0 | 0 | 5 | 2 | friel_masters |
| standard | build | 1 | 2 | 0 | 999 | 15 | 2 | 1 | 0 | 0 | 6 | 2 | esteve_lanao |
| standard | build | 3 | 3 | 0 | 39 | 20 | 2 | 3 | 1 | 0 | 5 | 2 | esteve_lanao |
| standard | build | 3 | 3 | 40 | 54 | 18 | 2 | 2 | 1 | 0 | 5 | 2 | friel_masters |
| standard | build | 3 | 3 | 55 | 999 | 15 | 2 | 1 | 0 | 0 | 5 | 3 | friel_masters |
| standard | build | 4 | 5 | 0 | 39 | 22 | 2 | 4 | 2 | 0 | 4 | 2 | esteve_lanao |
| standard | build | 4 | 5 | 40 | 999 | 18 | 2 | 3 | 1 | 0 | 4 | 2 | friel_masters |
| standard | peak | 1 | 2 | 0 | 999 | 18 | 1 | 2 | 1 | 0 | 5 | 2 | mujika_2003 |
| standard | peak | 3 | 3 | 0 | 39 | 22 | 1 | 2 | 2 | 1 | 4 | 2 | mujika_2003 |
| standard | peak | 3 | 3 | 40 | 999 | 18 | 1 | 2 | 1 | 1 | 4 | 2 | friel_masters |
| standard | peak | 4 | 5 | 0 | 39 | 25 | 1 | 3 | 2 | 2 | 3 | 2 | mujika_2003 |
| standard | peak | 4 | 5 | 40 | 999 | 20 | 1 | 2 | 1 | 1 | 4 | 2 | friel_masters |
| standard | taper | 1 | 5 | 0 | 39 | 20 | 0 | 2 | 1 | 0 | 3 | 2 | mujika_2003 |
| standard | taper | 1 | 5 | 40 | 999 | 18 | 0 | 1 | 1 | 0 | 3 | 2 | friel_masters |
| standard | recovery | 1 | 5 | 0 | 999 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | seiler_2010 |
| micro_structured | base | 1 | 2 | 0 | 999 | 8 | 1 | 0 | 0 | 0 | 4 | 2 | micro_adapted |
| micro_structured | base | 3 | 3 | 0 | 39 | 12 | 2 | 1 | 0 | 0 | 4 | 2 | micro_adapted |
| micro_structured | base | 3 | 3 | 40 | 54 | 10 | 2 | 1 | 0 | 0 | 4 | 2 | micro_adapted |
| micro_structured | base | 3 | 3 | 55 | 999 | 8 | 1 | 0 | 0 | 0 | 4 | 3 | micro_adapted |
| micro_structured | base | 4 | 5 | 0 | 39 | 12 | 2 | 1 | 0 | 0 | 3 | 2 | micro_adapted |
| micro_structured | base | 4 | 5 | 40 | 999 | 10 | 2 | 1 | 0 | 0 | 3 | 2 | micro_adapted |
| micro_structured | build | 1 | 2 | 0 | 999 | 12 | 1 | 1 | 0 | 0 | 4 | 2 | micro_adapted |
| micro_structured | build | 3 | 3 | 0 | 39 | 15 | 2 | 2 | 1 | 0 | 3 | 2 | micro_adapted |
| micro_structured | build | 3 | 3 | 40 | 54 | 12 | 2 | 1 | 1 | 0 | 3 | 2 | micro_adapted |
| micro_structured | build | 3 | 3 | 55 | 999 | 10 | 1 | 1 | 0 | 0 | 4 | 3 | micro_adapted |
| micro_structured | build | 4 | 5 | 0 | 39 | 18 | 2 | 2 | 1 | 0 | 3 | 2 | micro_adapted |
| micro_structured | build | 4 | 5 | 40 | 999 | 15 | 2 | 2 | 1 | 0 | 3 | 2 | micro_adapted |
| micro_structured | peak | 1 | 2 | 0 | 999 | 15 | 1 | 1 | 1 | 0 | 3 | 2 | micro_adapted |
| micro_structured | peak | 3 | 3 | 0 | 39 | 18 | 1 | 2 | 1 | 0 | 3 | 2 | micro_adapted |
| micro_structured | peak | 3 | 3 | 40 | 999 | 15 | 1 | 1 | 1 | 0 | 3 | 2 | micro_adapted |
| micro_structured | peak | 4 | 5 | 0 | 39 | 20 | 1 | 2 | 1 | 1 | 3 | 2 | micro_adapted |
| micro_structured | peak | 4 | 5 | 40 | 999 | 18 | 1 | 2 | 1 | 0 | 3 | 2 | micro_adapted |
| micro_structured | taper | 1 | 5 | 0 | 39 | 15 | 0 | 1 | 1 | 0 | 3 | 2 | micro_adapted |
| micro_structured | taper | 1 | 5 | 40 | 999 | 12 | 0 | 1 | 0 | 0 | 3 | 2 | micro_adapted |
| micro_structured | recovery | 1 | 5 | 0 | 999 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | micro_adapted |
| all | base | 1 | 5 | 0 | 999 | 12 | 2 | 1 | 0 | 0 | 5 | 2 | fallback |
| all | build | 1 | 5 | 0 | 999 | 18 | 2 | 2 | 1 | 0 | 4 | 2 | fallback |
| all | peak | 1 | 5 | 0 | 999 | 20 | 1 | 2 | 1 | 1 | 4 | 2 | fallback |
| all | taper | 1 | 5 | 0 | 999 | 18 | 0 | 1 | 1 | 0 | 3 | 2 | fallback |
| all | recovery | 1 | 5 | 0 | 999 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | fallback |
<!-- /GATC -->

<!-- Source: session_budgets_v4.md -->

<!-- GATC:session_budgets_v4:min_slot_requirements -->
| zone | min_slot_minutes | rationale |
|------|------------------|-----------|
| R | 30 | Minimal recovery benefit below this |
| Z1 | 30 | MED for peripheral adaptation |
| Z2 | 40 | MED for mitochondrial density |
| Z3 | 55 | Warmup + intervals + cooldown |
| Z4 | 60 | Warmup + 4x4 + cooldown |
| Z5 | 60 | Warmup + vo2max intervals + cooldown |
| Z6 | 50 | Warmup + anaerobic work + cooldown |
| Z7 | 50 | Warmup + power work + cooldown |
| Z8 | 45 | Warmup + sprints + cooldown |
<!-- /GATC -->

<!-- Source: session_budgets_v4.md -->

<!-- GATC:session_budgets_v4:quality_priority -->
| priority | zone | rationale |
|----------|------|-----------|
| 1 | Z5 | Highest recovery cost, needs biggest slots |
| 2 | Z6 | Anaerobic capacity |
| 3 | Z7 | Max power |
| 4 | Z8 | Sprint/neuromuscular |
| 5 | Z4 | VO2max/threshold |
| 6 | Z3 | Tempo/lactate clearance |
<!-- /GATC -->

<!-- Source: session_budgets_v4.md -->

## session_structure_variants_v4 Tables

<!-- GATC:session_structure_variants_v4:fallback_variants -->
| zone | fallback_variant_id | rationale |
|------|---------------------|-----------|
| R | r_classic | Standard recovery movement |
| Z1 | z1_classic | Standard easy session |
| Z2 | z2_classic | Standard aerobic session |
| Z3 | z3_classic | Safe default tempo structure |
| Z4 | z4_classic | Norwegian standard |
| Z5 | z5_classic | Proven VO2max structure |
| Z6 | z6_classic | Standard anaerobic |
| Z7 | z7_classic | Standard max power |
| Z8 | z8_classic | Standard sprint |
<!-- /GATC -->

<!-- Source: session_structure_variants_v4.md -->

<!-- GATC:session_structure_variants_v4:zone_variants -->
| zone | variant_id | variant_name | pattern | work_min_default | work_min_min | work_min_max | rest_ratio | reps_min | reps_max | phases_allowed | level_min | level_max | priority | label_template |
|------|------------|--------------|---------|------------------|--------------|--------------|------------|----------|----------|----------------|-----------|-----------|----------|----------------|
| R | r_classic | Easy Movement | continuous | 20 | 15 | 45 | 0.0 | 1 | 1 | all | beginner | elite | 1 | {work}min easy movement |
| R | r_active_flush | Active Flush | continuous | 25 | 20 | 35 | 0.0 | 1 | 1 | all | intermediate | elite | 2 | {work}min active flush (low cadence) |
| Z1 | z1_classic | Steady Easy | continuous | 45 | 30 | 90 | 0.0 | 1 | 1 | all | beginner | elite | 1 | {work}min @Z1 steady |
| Z1 | z1_long | Long Easy | continuous | 90 | 60 | 180 | 0.0 | 1 | 1 | base,build | intermediate | elite | 2 | {work}min @Z1 long |
| Z1 | z1_extended | Extended Aerobic | continuous | 180 | 120 | 240 | 0.0 | 1 | 1 | base | advanced | elite | 3 | {work}min @Z1 extended |
| Z2 | z2_classic | Steady Aerobic | continuous | 50 | 40 | 90 | 0.0 | 1 | 1 | all | beginner | elite | 1 | {work}min @Z2 steady |
| Z2 | z2_long | Long Aerobic | continuous | 90 | 60 | 120 | 0.0 | 1 | 1 | base,build | intermediate | elite | 2 | {work}min @Z2 long |
| Z2 | z2_progressive | Progressive Aerobic | progressive | 60 | 45 | 90 | 0.0 | 1 | 1 | base,build | novice | elite | 3 | {work}min @Z2 progressive (Z1→Z2) |
| Z3 | z3_classic | Classic Tempo | equal | 12 | 8 | 20 | 0.25 | 2 | 4 | all | novice | elite | 1 | {reps}×{work}min @Z3 / {rest}min rest |
| Z3 | z3_cruise | Cruise Intervals | equal | 20 | 15 | 30 | 0.17 | 2 | 3 | base,build | intermediate | elite | 2 | {reps}×{work}min @Z3 cruise |
| Z3 | z3_continuous | Continuous Tempo | continuous | 30 | 20 | 45 | 0.0 | 1 | 1 | base | intermediate | elite | 3 | {work}min @Z3 continuous |
| Z3 | z3_big_block | Big Block Tempo | equal | 20 | 18 | 25 | 0.15 | 2 | 2 | base | advanced | elite | 4 | 2×{work}min @Z3 big blocks |
| Z4 | z4_classic | Classic Threshold | equal | 10 | 8 | 12 | 0.4 | 2 | 4 | all | novice | elite | 1 | {reps}×{work}min @Z4 / {rest}min rest |
| Z4 | z4_long | Long Threshold | equal | 15 | 12 | 20 | 0.33 | 2 | 3 | build,peak | intermediate | elite | 2 | {reps}×{work}min @Z4 / {rest}min rest |
| Z4 | z4_short | Short Threshold | equal | 6 | 4 | 8 | 0.5 | 3 | 6 | build,peak | novice | elite | 3 | {reps}×{work}min @Z4 / {rest}min rest |
| Z4 | z4_over_under | Over-Unders | alternating | 8 | 6 | 10 | 0.25 | 2 | 4 | build,peak | intermediate | elite | 4 | {reps}×{work}min over-under (2min >Z4 / 2min <Z4) |
| Z4 | z4_norwegian | Norwegian Long | equal | 16 | 14 | 18 | 0.25 | 4 | 6 | build,peak | advanced | elite | 5 | {reps}×{work}min @Z4 Norwegian style |
| Z5 | z5_classic | Classic VO2max | equal | 4 | 3 | 5 | 0.75 | 3 | 6 | build,peak | novice | elite | 1 | {reps}×{work}min @Z5 / {rest}min rest |
| Z5 | z5_short | Short VO2max | equal | 3 | 2 | 4 | 1.0 | 4 | 8 | peak | intermediate | elite | 2 | {reps}×{work}min @Z5 / {rest}min rest |
| Z5 | z5_30_30 | 30/30 Intervals | micro | 0.5 | 0.5 | 0.5 | 1.0 | 10 | 20 | build,peak | intermediate | elite | 3 | {reps}×30s @Z5 / 30s rest |
| Z5 | z5_30_15 | Rønnestad 30/15 | micro | 0.5 | 0.5 | 0.5 | 0.5 | 13 | 13 | peak | advanced | elite | 4 | 3 sets × 13×30s @Z5 / 15s rest (3min between sets) |
| Z5 | z5_long | Long VO2max | equal | 5 | 4 | 6 | 0.6 | 3 | 5 | build | advanced | elite | 5 | {reps}×{work}min @Z5 / {rest}min rest |
| Z5 | z5_40_20 | 40/20 Intervals | micro | 0.67 | 0.67 | 0.67 | 0.5 | 10 | 15 | peak | advanced | elite | 6 | {reps}×40s @Z5 / 20s rest |
| Z6 | z6_classic | Anaerobic Capacity | equal | 1.0 | 0.5 | 1.5 | 2.0 | 4 | 8 | peak | intermediate | elite | 1 | {reps}×{work_sec}s @Z6 / {rest}min rest |
| Z6 | z6_short | Short Anaerobic | equal | 0.5 | 0.5 | 0.75 | 3.0 | 6 | 10 | peak | novice | elite | 2 | {reps}×{work_sec}s @Z6 / {rest}min rest |
| Z6 | z6_long | Long Anaerobic | equal | 1.5 | 1.0 | 2.0 | 2.0 | 3 | 6 | peak | advanced | elite | 3 | {reps}×{work_sec}s @Z6 / {rest}min rest |
| Z6 | z6_descending | Descending Ladder | descending | 1.5 | 0.5 | 1.5 | 2.5 | 4 | 4 | peak | advanced | elite | 4 | 90s→60s→45s→30s @Z6 / full recovery |
| Z7 | z7_classic | Max Anaerobic | equal | 0.5 | 0.25 | 0.75 | 6.0 | 4 | 8 | peak | intermediate | elite | 1 | {reps}×{work_sec}s @Z7 / {rest}min rest |
| Z7 | z7_short | Short Max Power | equal | 0.25 | 0.17 | 0.33 | 8.0 | 6 | 10 | peak | novice | elite | 2 | {reps}×{work_sec}s @Z7 / {rest}min rest |
| Z7 | z7_cluster | Cluster Sets | cluster | 0.33 | 0.33 | 0.33 | 6.0 | 9 | 9 | peak | advanced | elite | 3 | 3×3×20s @Z7 cluster (short rest in cluster, long between) |
| Z8 | z8_classic | Neuromuscular Sprint | equal | 0.17 | 0.1 | 0.25 | 20.0 | 6 | 10 | all | novice | elite | 1 | {reps}×{work_sec}s @Z8 sprint / {rest}min rest |
| Z8 | z8_flying | Flying Sprints | equal | 0.12 | 0.08 | 0.17 | 25.0 | 8 | 12 | peak | intermediate | elite | 2 | {reps}×{work_sec}s flying sprint |
| Z8 | z8_strides | Strides | equal | 0.17 | 0.15 | 0.25 | 15.0 | 4 | 6 | all | beginner | elite | 3 | {reps}×{work_sec}s strides |
<!-- /GATC -->

<!-- Source: session_structure_variants_v4.md -->

## session_templates_v4 Tables

<!-- GATC:session_templates_v4:warmup_scaling -->
| zone_type | container_min | container_max | warmup_pct | warmup_max_min | cooldown_pct | cooldown_max_min |
|-----------|---------------|---------------|------------|----------------|--------------|------------------|
| easy | 0 | 999 | 0.00 | 0 | 0.00 | 0 |
| intensity | 0 | 44 | 0.15 | 5 | 0.10 | 5 |
| intensity | 45 | 59 | 0.20 | 10 | 0.10 | 5 |
| intensity | 60 | 999 | 0.25 | 15 | 0.15 | 10 |
<!-- /GATC -->

<!-- Source: session_templates_v4.md -->

## training_load_model_v4 Tables

<!-- GATC:training_load_model_v4:adaptation_rate -->
| training_years_min | training_years_max | age_min | age_max | vo2max_gain_pct | endurance_gain_pct | threshold_gain_pct | growth_rate | decay_rate | source | confidence |
|--------------------|--------------------|---------|---------|-----------------|--------------------|--------------------| ------------|------------|--------|------------|
| 0 | 0.5 | 0 | 39 | 4.0 | 5.0 | 4.5 | 0.10 | 0.08 | foster_seiler | high |
| 0 | 0.5 | 40 | 54 | 3.2 | 4.0 | 3.6 | 0.09 | 0.09 | tanaka_2008 | high |
| 0 | 0.5 | 55 | 64 | 2.4 | 3.0 | 2.7 | 0.08 | 0.10 | tanaka_2008 | medium |
| 0 | 0.5 | 65 | 999 | 1.8 | 2.2 | 2.0 | 0.07 | 0.11 | expert_consensus | low |
| 0.5 | 2 | 0 | 39 | 2.0 | 2.5 | 2.2 | 0.09 | 0.09 | banister_busso | high |
| 0.5 | 2 | 40 | 54 | 1.6 | 2.0 | 1.8 | 0.08 | 0.10 | tanaka_2008 | high |
| 0.5 | 2 | 55 | 64 | 1.2 | 1.5 | 1.3 | 0.07 | 0.11 | tanaka_2008 | medium |
| 0.5 | 2 | 65 | 999 | 0.9 | 1.1 | 1.0 | 0.06 | 0.12 | expert_consensus | low |
| 2 | 5 | 0 | 39 | 1.0 | 1.3 | 1.1 | 0.08 | 0.10 | seiler_2010 | high |
| 2 | 5 | 40 | 54 | 0.8 | 1.0 | 0.9 | 0.07 | 0.11 | tanaka_2008 | medium |
| 2 | 5 | 55 | 64 | 0.6 | 0.8 | 0.7 | 0.06 | 0.12 | tanaka_2008 | medium |
| 2 | 5 | 65 | 999 | 0.5 | 0.6 | 0.5 | 0.05 | 0.13 | expert_consensus | low |
| 5 | 10 | 0 | 39 | 0.5 | 0.7 | 0.6 | 0.07 | 0.11 | mujika_2003 | medium |
| 5 | 10 | 40 | 54 | 0.4 | 0.5 | 0.45 | 0.06 | 0.12 | expert_consensus | low |
| 5 | 10 | 55 | 64 | 0.3 | 0.4 | 0.35 | 0.05 | 0.13 | expert_consensus | low |
| 5 | 10 | 65 | 999 | 0.25 | 0.3 | 0.27 | 0.05 | 0.14 | expert_consensus | low |
| 10 | 999 | 0 | 39 | 0.3 | 0.4 | 0.35 | 0.06 | 0.12 | mujika_2003 | medium |
| 10 | 999 | 40 | 54 | 0.25 | 0.3 | 0.27 | 0.05 | 0.13 | expert_consensus | low |
| 10 | 999 | 55 | 64 | 0.2 | 0.25 | 0.22 | 0.05 | 0.14 | expert_consensus | low |
| 10 | 999 | 65 | 999 | 0.15 | 0.2 | 0.17 | 0.04 | 0.15 | expert_consensus | low |
<!-- /GATC -->

<!-- Source: training_load_model_v4.md -->

<!-- GATC:training_load_model_v4:frequency_templates -->
| sessions_per_7d | pattern | notes |
|-----------------|---------|-------|
| 2 | H,OFF,OFF,M,OFF,OFF,OFF | Time-limited, max recovery |
| 3 | H,OFF,M,OFF,H,OFF,OFF | Classic hard-easy |
| 4 | H,OFF,M,OFF,H,L,OFF | Add easy day |
| 5 | M,H,OFF,M,H,L,OFF | Balanced with variety |
| 6 | M,H,L,M,H,L,OFF | Polarized friendly |
| 7 | M,H,L,M,H,L,R | Full week, enforce recovery |
<!-- /GATC -->

<!-- Source: training_load_model_v4.md -->

<!-- GATC:training_load_model_v4:load_ceiling -->
| training_years_min | training_years_max | age_min | age_max | weekly_hours_max | weekly_uls_max | meso_ceiling_mult | ramp_max_pct_per_meso | source | confidence |
|--------------------|--------------------|---------|---------| -----------------|----------------|-------------------|----------------------|--------|------------|
| 0 | 0.5 | 0 | 39 | 4 | 300 | 1.10 | 8 | foster_2001 | high |
| 0 | 0.5 | 40 | 54 | 3.5 | 260 | 1.08 | 6 | tanaka_2008 | high |
| 0 | 0.5 | 55 | 64 | 3 | 220 | 1.06 | 5 | expert_consensus | medium |
| 0 | 0.5 | 65 | 999 | 2.5 | 180 | 1.05 | 4 | expert_consensus | low |
| 0.5 | 2 | 0 | 39 | 8 | 550 | 1.15 | 10 | foster_2001 | high |
| 0.5 | 2 | 40 | 54 | 7 | 480 | 1.12 | 8 | tanaka_2008 | high |
| 0.5 | 2 | 55 | 64 | 6 | 400 | 1.10 | 7 | expert_consensus | medium |
| 0.5 | 2 | 65 | 999 | 5 | 350 | 1.08 | 6 | expert_consensus | low |
| 2 | 5 | 0 | 39 | 12 | 800 | 1.18 | 12 | seiler_2010 | high |
| 2 | 5 | 40 | 54 | 10 | 680 | 1.15 | 10 | tanaka_2008 | medium |
| 2 | 5 | 55 | 64 | 8 | 550 | 1.12 | 8 | expert_consensus | medium |
| 2 | 5 | 65 | 999 | 7 | 480 | 1.10 | 7 | expert_consensus | low |
| 5 | 10 | 0 | 39 | 18 | 1100 | 1.20 | 15 | seiler_2010 | medium |
| 5 | 10 | 40 | 54 | 15 | 950 | 1.18 | 12 | expert_consensus | low |
| 5 | 10 | 55 | 64 | 12 | 780 | 1.15 | 10 | expert_consensus | low |
| 5 | 10 | 65 | 999 | 10 | 650 | 1.12 | 8 | expert_consensus | low |
| 10 | 999 | 0 | 39 | 25 | 1500 | 1.22 | 15 | tonnessen_2014 | medium |
| 10 | 999 | 40 | 54 | 20 | 1250 | 1.20 | 12 | expert_consensus | low |
| 10 | 999 | 55 | 64 | 16 | 1000 | 1.17 | 10 | expert_consensus | low |
| 10 | 999 | 65 | 999 | 14 | 850 | 1.15 | 8 | expert_consensus | low |
<!-- /GATC -->

<!-- Source: training_load_model_v4.md -->

<!-- GATC:training_load_model_v4:zone_duration_thresholds -->
| zone | threshold_min | md_cap | rationale |
|------|---------------|--------|-----------|
| R | 90 | 1.40 | Recovery, minimal glycogen demand |
| Z1 | 90 | 1.50 | Very low intensity, sustainable |
| Z2 | 60 | 1.60 | Aerobic base, glycogen depletion begins |
| Z3 | 45 | 1.50 | Tempo, lactate accumulation |
| Z4 | 30 | 1.45 | Threshold, significant metabolic stress |
| Z5 | 20 | 1.40 | VO2max, rapid fatigue accumulation |
| Z6 | 20 | 1.35 | Anaerobic capacity, acute stress |
| Z7 | 20 | 1.30 | Neuromuscular, CNS fatigue |
| Z8 | 20 | 1.30 | Sprint, CNS fatigue |
<!-- /GATC -->

<!-- Source: training_load_model_v4.md -->

---

## Hard Boundaries

This card does NOT:
- Define training policy (see axis cards)
- Compute session load (see session_rules.md)
- Define zone meanings (see zone_physiology.md)
- Define readiness states (see load_monitoring.md)

This card contains GATC operational parameters only.

End of card.
