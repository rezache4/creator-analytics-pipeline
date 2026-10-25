# Comment Features — Data Dictionary

Output file: `video_comment_features.parquet`  
Granularity: one row per YouTube video (1,076 videos across 5 cohort channels)

## Identifiers
| Feature | Description |
|---|---|
| `media_id` | YouTube video ID |
| `channelTitle` | Channel name |
| `n_comments` | Number of comments analysed for this video |

## Sentiment Signals
| Feature | Description |
|---|---|
| `mean_score` | Mean LLM sentiment score across all comments (−1 very negative → +1 very positive) |
| `score_std` | Standard deviation of sentiment scores — how spread opinion is |
| `pos_frac` | Share of comments labelled positive |
| `neg_frac` | Share of comments labelled negative |
| `neu_frac` | Share of comments labelled neutral |
| `score_slope` | Linear trend of sentiment score over comment timestamps (per second); negative = audience sentiment sours as more comments arrive |
| `early_late_delta` | Mean score of the first 20% of comments minus the last 20% by time; positive = early commenters were more positive than late ones |
| `vibe_score` | Per-video mean sentiment score from the room_vibe table (authoritative, computed over all comments including those not in a sentiment chunk) |
| `dispersion` | Std of scores from room_vibe — same concept as `score_std` but from the room computation pass |

## Audience Dynamics
| Feature | Description |
|---|---|
| `stance_alignment` | Mean sign of scores (−1 = all commenters against creator/post, +1 = all for) |
| `polarization` | 2 × min(pos_frac, neg_frac); peaks at 1 when the audience is evenly split positive/negative |
| `controversy` | 4 × pos_frac × neg_frac; peaks when pos≈neg, zero when one side dominates |
| `consensus` | 1 − normalised entropy of {pos, neg, neu} label mix; 1 = everyone gives the same label |
| `sarcasm_rate` | Share of comments flagged as sarcastic by the LLM |
| `toxicity_rate` | Share of comments with toxicity ≠ "none" |
| `room_state` | Categorical classification of the video's comment section: `united` / `mixed` / `divided` |

## LLM Room Read (narrative, one per video)
| Feature | Description |
|---|---|
| `llm_vibe` | LLM's overall vibe label for the comment section (e.g. celebratory, critical, divided) |
| `llm_consensus` | LLM's 0–1 consensus estimate |
| `llm_alignment` | LLM's −1..+1 alignment of commenters toward the creator |
| `llm_controversy` | LLM's 0–1 controversy estimate |
| `dominant_stance` | Short phrase describing the majority commenter position |
| `split_axis` | What divides the room (null if united) |
| `room_summary` | 1–2 sentence narrative of the comment section vibe |

## Intent
| Feature | Description |
|---|---|
| `praise_rate` | Share of comments with intent = praise |
| `criticism_rate` | Share of comments with intent = criticism |

## Emotion
| Feature | Description |
|---|---|
| `top_emotion` | Most frequent emotion label among the video's comments |
| `emotion_entropy` | Normalised Shannon entropy of the emotion distribution; 0 = one emotion dominates, 1 = evenly spread across all emotions |
