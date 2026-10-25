rm(list=ls())
library(settings)
reset(options)
graphics.off()

library(nlmeU)  # --> for the dataset
library(nlme)   # --> for models implementation
library(lattice)
library(corrplot)
library(plot.matrix)
library(visdat)
library(car)
library(ggplot2)
library(lme4)
library(insight)
library(RLRsim) 
library(dplyr)
library(lubridate)
library(forcats)
library(caret)
library(randomForest)
library(GGally)
library(tidyr)
library(glmmTMB)
library(DHARMa)

df <- read.csv(
  "C1_descriptive_model_df_v4.csv",
  colClasses = c(post_id = "character")
)

head(df)
df$post_id <- as.factor(df$post_id)
df$is_weekend <- as.logical(df$is_weekend)
df$hour_band <- as.factor(df$hour_band)
df$media_type <- as.factor(df$media_type)

df$mentions_alice <- as.logical(df$mentions_alice)
df$mentions_guglielmo <- as.logical(df$mentions_guglielmo)
df$mentions_aimone <- as.logical(df$mentions_aimone)
df$mentions_claudio <- as.logical(df$mentions_claudio)
df$mentions_creators <- as.logical(df$mentions_creators)
df$mentions_famous <- as.logical(df$mentions_famous)
df$mentions_singer <- as.logical(df$mentions_singer)
df$mentions_fashion <- as.logical(df$mentions_fashion)
df$has_mentions <- as.logical(df$has_mentions)
df$mentions_brand <- as.logical(df$mentions_brand)
df$mentions_other_people <- as.logical(df$mentions_other_people)
df$is_adv <- as.logical(df$is_adv)
df$is_supplied <- as.logical(df$is_supplied)
df$is_gifted <- as.logical(df$is_gifted)
df$is_sanremo <- as.logical(df$is_sanremo)
df$topic <- as.factor(df$topic)
df$has_love_emoji <- as.logical(df$has_love_emoji)
df$has_lol_emoji <- as.logical(df$has_lol_emoji)
df$has_shine_emoji <- as.logical(df$has_shine_emoji)
df$has_call_to_action <- as.logical(df$has_call_to_action)
df$is_ironic <- as.logical(df$is_ironic)
df$audio_type <- as.factor(df$audio_type)
df$location <- as.factor(df$location)

df$is_video <- ifelse(is.na(df$video_duration), 0, 1)

df$video_duration_filled <- ifelse(
  is.na(df$video_duration),
  0,
  df$video_duration
)

df$n_hashtags[is.na(df$n_hashtags)] <- 0
df$n_mentions[is.na(df$n_mentions)] <- 0

df <- df %>%
  mutate(
    section = case_when(
      media_type %in% c("CAROUSEL", "IMAGE") ~ "FEED",
      media_type == "VIDEO_REELS" ~ "REELS",
      TRUE ~ NA_character_
    )
  )

df$section <- as.factor(df$section)


df <- df %>%
  mutate(
    has_close_friend = ifelse(
      mentions_alice | mentions_guglielmo | mentions_claudio,
      1, 0
    )
  )

# Create year factor
df <- df %>%
  mutate(
    year_f = factor(
      format(as.POSIXct(as.character(timestamp), format = "%Y-%m-%d %H:%M:%S"), "%Y"),
      levels = c("2023", "2024", "2025", "2026")
    )
  )

# Create quadrimester factor
ts <- as.POSIXct(as.character(df$timestamp), format = "%Y-%m-%d %H:%M:%S")
yr <- format(ts, "%Y")
mn <- as.integer(format(ts, "%m"))

quad <- ifelse(mn <= 4, "t1",
               ifelse(mn <= 8, "t2", "t3"))

df$quadrimester_f <- paste(quad, yr)

# Set quadrimester factor levels in chronological order
df$quadrimester_f <- factor(
  df$quadrimester_f,
  levels = c(
    "t1 2023", "t2 2023", "t3 2023",
    "t1 2024", "t2 2024", "t3 2024",
    "t1 2025", "t2 2025", "t3 2025", "t1 2026"
  )
)



df <- df %>%
  mutate(
    season = case_when(
      month %in% c(12, 1, 2) ~ "winter",
      month %in% c(3, 4, 5) ~ "spring",
      month %in% c(6, 7, 8) ~ "summer",
      month %in% c(9, 10, 11) ~ "autumn",
      TRUE ~ NA_character_
    ),
    
    season_year = case_when(
      month == 12 ~ year + 1,
      TRUE ~ year
    ),
    
    season = factor(
      season,
      levels = c("winter","spring", "summer", "autumn" )
    )
  )


df <- df %>%
  arrange(timestamp) %>%
  mutate(post_number = row_number())

df$hour <- as.factor(df$hour)
df$month <- as.factor(df$month)

library(fastDummies)

df_rf<- df %>%
  dummy_cols(
    select_columns = "topic",
    remove_first_dummy = FALSE,
    remove_selected_columns = TRUE
  )

topic_dummy_vars <- grep("^topic_", names(df_rf), value = TRUE)

df_rf[topic_dummy_vars] <- lapply(df_rf[topic_dummy_vars], as.logical)
names(df_rf) <- gsub("&", "and", names(df_rf))
names(df_rf) <- gsub("[^A-Za-z0-9_]", "_", names(df_rf))
names(df_rf) <- gsub("_+", "_", names(df_rf))
names(df_rf) <- gsub("_$", "", names(df_rf))
##### Train-test split ####
set.seed(123)
train_idx <- createDataPartition(df_rf$share_C1_pre, p = 0.7, list = FALSE)

train_df <- df_rf[train_idx, ]
test_df  <- df_rf[-train_idx, ]


# Define formula (target ~ features)
rf_formula <- share_C1_pre ~ year + month + is_weekend + hour_band + hour + media_type+
  mentions_alice+ mentions_guglielmo+ mentions_aimone+ mentions_claudio +
  mentions_famous+ mentions_creators + mentions_singer + mentions_fashion+
  mentions_brand + mentions_other_people + is_adv + 
  is_supplied + is_gifted + is_sanremo +caption_length +avg_words_per_sentence+
  exclamation_marks + question_marks + has_love_emoji + has_lol_emoji + has_shine_emoji+
  has_call_to_action + posting_frequency_last7d + days_since_last_post + days_since_last_adv +
  adv_density_last30d + same_topic_last_10posts+ is_ironic+ audio_type + location+
  n_hashtags  + video_duration_filled +section +season +topic_Beauty_and_Fashion + topic_Cinema_and_TV+
  topic_Community + topic_Cooking + topic_Culture + topic_Food + topic_Games +
  topic_Other + topic_Podcast + topic_Private_Life +
  topic_Reflections + topic_Relatable_Comedy + topic_Theatre +
  topic_Travel + topic_YouTube

# Train RF
set.seed(123)
rf_model <- randomForest(
  formula = rf_formula,
  data    = train_df,
  ntree   = 300,
  mtry    = 4,
  importance = TRUE
)

print(rf_model)

# Show importance table
importance(rf_model)

# Plot variable importance
varImpPlot(
  rf_model,
  main = "Random Forest - Variable Importance",
  n.var = 30,            # show top 
  type = 1               # type=1 → mean decrease in accuracy
)


#### Correlation ####

num_vars <- c(
  "share_C1_pre", "year", "month", "hour",
  "caption_length", "avg_words_per_sentence",
  "exclamation_marks", "question_marks",
  "posting_frequency_last7d", "days_since_last_post",
  "days_since_last_adv", "adv_density_last30d",
  "same_topic_last_10posts", "n_hashtags",
  "video_duration_filled"
)

df_num <- df[, num_vars]


GGally::ggcorr(df_num, label = TRUE, hjust = 0.9, size = 4)

ggpairs(df_num)


##### EDA ####

# Histogram of c1 percentage 
ggplot(df, aes(x = share_C1_pre)) +
  geom_histogram(bins = 50, alpha = 0.6) +
  labs(
    title = "C1_share_prev Histogram"
  ) +
  theme_minimal()

# Histogram of c1 percentage by media type
ggplot(df, aes(x = share_C1_pre, fill = section)) +
  geom_histogram(bins = 50, alpha = 0.6) +
  facet_wrap(~section) +
  theme_minimal()


# Boxplot of c1 percentage by section

ggplot(df, aes(share_C1_pre, fill = section)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(title = "C1_prev_month % by section")

###### Friends boxplots ######
# Boxplot of c1 percentage by has_close_friend
ggplot(df, aes(
  y = factor(has_close_friend, labels = c("No", "Yes")),
  x = share_C1_pre,
  fill = factor(has_close_friend, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_prev_month % by has_close_friend",
    y = "Has Close friend",
    x = "C1_prev_month percentage",
    fill = "Has close friend"
  )

# Boxplot of c1 percentage by mentions_guglielmo
ggplot(df, aes(
  y = factor(mentions_guglielmo, labels = c("No", "Yes")),
  x = share_C1_pre,
  fill = factor(mentions_guglielmo, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_prev_month % by mentions_guglielmo",
    y = "Mentions Guglielmo",
    x = "C1_prev_month percentage",
    fill = "Mentions Guglielmo"
  )

# Boxplot of c1 percentage by mentions_alice
ggplot(df, aes(
  y = factor(mentions_alice, labels = c("No", "Yes")),
  x = share_C1_pre,
  fill = factor(mentions_alice, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_prev_month % by mentions_alice",
    y = "Mentions Alice",
    x = "C1_prev_month percentage",
    fill = "Mentions Alice"
  )

# Boxplot of c1 percentage by mentions_claudio
ggplot(df, aes(
  y = factor(mentions_claudio, labels = c("No", "Yes")),
  x = share_C1_pre,
  fill = factor(mentions_claudio, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_prev_month % by mentions_claudio",
    y = "Mentions claudio",
    x = "C1_prev_month percentage",
    fill = "Mentions claudio"
  )

# Boxplot of c1 percentage by mentions_aimone
ggplot(df, aes(
  y = factor(mentions_aimone, labels = c("No", "Yes")),
  x = share_C1_pre,
  fill = factor(mentions_aimone, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_prev_month % by mentions_aimone",
    y = "Mentions aimone",
    x = "C1_prev_month percentage",
    fill = "Mentions aimone"
  )

###### Boxplot of posts characteristics ######

# Boxplot of c1 percentage by is_adv
ggplot(df, aes(
  y = factor(is_adv, labels = c("No", "Yes")),
  x = share_C1_pre,
  fill = factor(is_adv, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_prev_month % by is_adv",
    y = "Is_adv",
    x = "C1_prev_month percentage",
    fill = "Is_adv"
  )

# Boxplot of c1 percentage by is_sanremo
ggplot(df, aes(
  y = factor(is_sanremo, labels = c("No", "Yes")),
  x = share_C1_pre,
  fill = factor(is_sanremo, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_prev_month % by is_sanremo",
    y = "Is_sanremo",
    x = "C1_prev_month percentage",
    fill = "Is_sanremo"
  )

# Boxplot of c1 percentage by topic (excluding Uncertain, Dance and Other)
ggplot(
  subset(df, !topic %in% c("Uncertain", "Dance", "Other")),
  aes(share_C1_pre, fill = topic)
) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(title = "C1_prev_month % by topic")

# Boxplot of c1 percentage by location
ggplot(df, aes(
  y = factor(location, labels = c("fake", "no", "real")),
  x = share_C1_pre,
  fill = factor(location, labels = c("fake", "no", "real"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_prev_month % by location",
    y = "Location",
    x = "C1_prev_month percentage",
    fill = "Location"
  )

# Boxplot of c1 percentage by audio_type
ggplot(df, aes(
  y = factor(audio_type, labels = c("licensed", "original", "no")),
  x = share_C1_pre,
  fill = factor(audio_type, labels = c("licensed", "original", "no"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_prev_month % by audio_type",
    y = "Audio type",
    x = "C1_prev_month percentage",
    fill = "audio type"
  )

# Boxplot of c1 percentage by has_call_to_action
ggplot(df, aes(
  y = factor(has_call_to_action),
  x = share_C1_pre,
  fill = factor(has_call_to_action)
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_prev_month % by has_call_to_action",
    y = "has_call_to_action",
    x = "C1_prev_month percentage",
    fill = "has_call_to_action"
  )

# Boxplot of c1 percentage by has_lol_emoji
ggplot(df, aes(
  y = factor(has_lol_emoji),
  x = share_C1_pre,
  fill = factor(has_lol_emoji)
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_prev_month % by has_lol_emoji",
    y = "has_lol_emoji",
    x = "C1_prev_month percentage",
    fill = "has_lol_emoji"
  )

# Boxplot of c1 percentage by has_shine_emoji
ggplot(df, aes(
  y = factor(has_shine_emoji),
  x = share_C1_pre,
  fill = factor(has_shine_emoji)
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_prev_month % by has_shine_emoji",
    y = "has_shine_emoji",
    x = "C1_prev_month percentage",
    fill = "has_shine_emoji"
  )

###### Boxplot of posts time settings ######

# Boxplot of c1 percentage by hour_band
ggplot(df, aes(
  y = factor(hour_band, labels = c("afternoon", "evening", "morning")),
  x = share_C1_pre,
  fill = factor(hour_band, labels = c("afternoon", "evening", "morning"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_prev_month % by hour_band",
    y = "Hour_band",
    x = "C1_prev_month percentage",
    fill = "hour band"
  )



# Boxplot C1_prev_month % by season and year
ggplot(df, aes(
  x = season,
  y = share_C1_pre,
  fill = season
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_wrap(~ season_year, nrow = 1) +
  theme_minimal() +
  labs(
    title = "C1_prev_month % by season and year",
    x = "Season",
    y = "C1_prev_month percentage",
    fill = "Season"
  )

ggplot(df, aes(
  x = season,
  y = share_C1_pre,
  fill = season
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_grid(section ~ season_year) +
  theme_minimal() +
  labs(
    title = "C1_prev_month % by season, year and section",
    x = "Season",
    y = "C1_prev_month percentage",
    fill = "Season"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom"
  )

# Boxplot of c1 percentage by year and section
bw1 <- bwplot(share_C1_pre~ year_f | section, data = df)
xlims <- c("2023","2024","2025", "2026")
update(bw1, xlim = xlims, pch = "|")


# Boxplot of c1 percentage by year and section
bw2 <- bwplot(share_C1_pre~ as.factor(month) | as.factor(year), data = df)
xlims <- c("gen","feb","mar", "apr", "may","jun","jul", "aug", "sep","oct","nov", "dec")
update(bw2, xlim = xlims, pch = "|")


# Boxplot of c1 percentage by year and section
bw3 <- bwplot(share_C1_pre~ season | as.factor(year), data = df)
xlims <- c("spring", "summer", "autumn", "winter")
update(bw3, xlim = xlims, pch = "|")

####### Video duration by topic ######
df %>%
  filter(video_duration_filled > 0, !topic %in% c("Other")) %>%
  ggplot(aes(x = topic, y = video_duration_filled, fill = topic)) +
  geom_boxplot() +
  labs(
    title = "Video duration by topic",
    x = "Topic",
    y = "Video Duration"
  ) +
  theme_minimal() +
  theme(legend.position = "none")


#### Beta-binomial model ####

library(glmmTMB)
train_df <- train_df %>%
  mutate(
    n_total = n_C1_pre + n_non_C1_pre,
    year = factor(year),
    month = factor(month),
    season = factor(season),
    hour_band = factor(hour_band),
    hour = factor(hour), 
    location = factor(location),
    section = factor(section),
    audio_type = factor(audio_type)
  ) %>%
  filter(n_total > 0)

test_df <- test_df %>%
  mutate(
    n_total = n_C1_pre + n_non_C1_pre,
    year = factor(year),
    month = factor(month),
    season = factor(season),
    hour_band = factor(hour_band),
    hour = factor(hour),
    location = factor(location),
    section = factor(section),
    audio_type = factor(audio_type)
  ) %>%
  filter(n_total > 0)

mod_bb_complex <- glmmTMB(
  cbind(n_C1_pre, n_non_C1_pre) ~
    location+
    year+
    topic_YouTube +
    adv_density_last30d+
    month +
    question_marks +
    video_duration_filled +
    posting_frequency_last7d +
    mentions_alice +
    avg_words_per_sentence +
    topic_Private_Life +
    section +
    mentions_fashion +
    mentions_creators +
    audio_type +
    mentions_brand +
    caption_length +
    exclamation_marks +
    mentions_famous +
    topic_Travel +
    same_topic_last_10posts +
    has_call_to_action +
    mentions_singer +
    mentions_guglielmo +
    hour +
    topic_Theatre, 
  family = betabinomial(),
  data = train_df
)

summary(mod_bb_complex)

# By substituting month with season and hour with hour_band we get an improvement in BIC, AIC gets slightly better
#AIC went from 2165 (mod_bb_complex) to 2163
# BIC from 2372 to 2295
mod_bb_simple <- glmmTMB(
  cbind(n_C1_pre, n_non_C1_pre) ~
    location+
    year+
    topic_YouTube +
    adv_density_last30d+
    season +
    question_marks +
    video_duration_filled +
    topic_Private_Life +
    mentions_creators +
    same_topic_last_10posts +
    mentions_singer, 
  family = betabinomial(),
  data = train_df
)

summary(mod_bb_simple)

# Removed variables proceeding stepwise:
# hour_band, topic_Theatre , mentions_guglielmo , has_call_to_action,topic_Travel,exclamation_marks,
# mentions_brand, audio_type, mentions_fashion, section, posting_frequency_last7d,
# mentions_alice, caption_length, avg_words_per_sentence, mentions_famous, 


mod_no_question <- update(
  mod_bb_simple,
  . ~ . - question_marks
)

anova(mod_no_question, mod_bb_simple)

# anova(mod_no_question, mod_bb_simple) gave a p-value around 0.056, so I removed question_marks
# AIC from 2146.9 to 2148.6
# BIC from 2214.7 to 2212.6

mod_full1 <- glmmTMB(
  cbind(n_C1_pre, n_non_C1_pre) ~
    location+
    year+
    topic_YouTube +
    adv_density_last30d+
    season +
    video_duration_filled +
    topic_Private_Life +
    mentions_creators +
    same_topic_last_10posts +
    mentions_singer, 
  family = betabinomial(),
  data = train_df
)

summary(mod_full1)

mod_no_adv_density <- update(
  mod_full1,
  . ~ . - adv_density_last30d
)

anova(mod_no_adv_density, mod_full1)
# anova(mod_no_days_since_last_adv, mod_full1) gave a p-value around 0.064, so I removed adv_density_last30d
# AIC from 2148.6 to 2150
# BIC from 2212.6 to 2210.2

mod_full2 <- glmmTMB(
  cbind(n_C1_pre, n_non_C1_pre) ~
    location+
    year+
    topic_YouTube+
    season +
    video_duration_filled +
    topic_Private_Life +
    mentions_creators +
    same_topic_last_10posts +
    mentions_singer, 
  family = betabinomial(),
  data = train_df
)

summary(mod_full2)

mod_final <- mod_full2

# Check for residuals
set.seed(123)

res_final <- simulateResiduals(
  fittedModel = mod_final,
  n = 1000
)

plot(res_final)

testUniformity(res_final)
testDispersion(res_final)
## UNDERDISPERSION DETECTED
testZeroInflation(res_final)
testOutliers(res_final)

library(performance)
check_collinearity(mod_final)

plotResiduals(res_final, form = train_df$video_duration_filled)
plotResiduals(res_final, form = train_df$same_topic_last_10posts)


train_df$share_obs <- train_df$n_C1_pre / 
  (train_df$n_C1_pre + train_df$n_non_C1_pre)

train_df$share_pred <- predict(mod_final, type = "response")

ggplot(train_df, aes(x = share_pred, y = share_obs)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  theme_minimal() +
  labs(
    title = "Observed vs predicted C1 share",
    x = "Predicted share C1",
    y = "Observed share C1"
  )

#### ODDS RATIO ####

coefs <- summary(mod_final)$coefficients$cond

or_table <- data.frame(
  term = rownames(coefs),
  estimate = coefs[, "Estimate"],
  OR = exp(coefs[, "Estimate"]),
  CI_low = exp(coefs[, "Estimate"] - 1.96 * coefs[, "Std. Error"]),
  CI_high = exp(coefs[, "Estimate"] + 1.96 * coefs[, "Std. Error"]),
  p_value = coefs[, "Pr(>|z|)"]
)

or_table

#OR > 1  = higher  odds that the commenter is a C1 (C1 from the former update)
#OR < 1  = lower  odds that the commenter is a C1


#### Spline trial ####
library(splines)

mod_spline_video <- update(
  mod_final,
  . ~ . - video_duration_filled + ns(video_duration_filled, df = 3)
)


mod_spline_same_topic <- update(
  mod_final,
  . ~ . - same_topic_last_10posts + ns(same_topic_last_10posts, df = 3)
)

mod_spline_both <- update(
  mod_final,
  . ~ . 
  - video_duration_filled 
  - same_topic_last_10posts +
    ns(video_duration_filled, df = 3) +
    ns(same_topic_last_10posts, df = 3)
)

AIC(
  mod_final,
  mod_spline_video,
  mod_spline_same_topic,
  mod_spline_both
)

BIC(
  mod_final,
  mod_spline_video,
  mod_spline_same_topic,
  mod_spline_both
)
# mod_final is still the best

#### Linear Model trial ####

train_df$n_total <- train_df$n_C1_pre + train_df$n_non_C1_pre

mod_gls1 <- gls(
  share_C1_pre~
    location+
    year+
    topic_YouTube +
    adv_density_last30d+
    season +
    question_marks +
    topic_Private_Life +
    mentions_creators +
    same_topic_last_10posts +
    mentions_singer,
  data = train_df
)
summary(mod_gls1)

plot(mod_gls1)

res_gls <- residuals(mod_gls1, type = "normalized")

acf(res_gls)

Box.test(res_gls, lag = 10, type = "Ljung-Box")
Box.test(res_gls, lag = 20, type = "Ljung-Box")

qqnorm(res_gls)
qqline(res_gls)
shapiro.test(res_gls)

# Normality is violated

mod_gls2 <- gls(
  share_C1_pre~
    location+
    year+
    topic_YouTube +
    adv_density_last30d+
    season +
    question_marks +
    topic_Private_Life +
    mentions_creators +
    same_topic_last_10posts +
    mentions_singer,
  weights = varIdent(form = ~1|year),
  data = train_df
)
summary(mod_gls2)

anova(mod_gls1, mod_gls2)

# Check for residuals of gls2

plot(mod_gls2)

qqnorm(resid(mod_gls2, type = "normalized"))
qqline(resid(mod_gls2, type = "normalized"))

acf(resid(mod_gls2, type = "normalized"))

Box.test(
  resid(mod_gls2, type = "normalized"),
  lag = 10,
  type = "Ljung-Box"
)

shapiro.test(resid(mod_gls2, type = "normalized"))


gls_coef <- as.data.frame(summary(mod_gls2)$tTable)

gls_coef$term <- rownames(gls_coef)
gls_coef$effect_percentage_points <- gls_coef$Value * 100

gls_coef


# Rebuild topic
# Crea tabella di lookup post_id -> topic dal dataset originale df
topic_lookup <- df %>%
  select(post_id, topic) %>%
  distinct(post_id, .keep_all = TRUE) %>%
  mutate(topic = factor(topic))

# Aggiungi topic a train_df
train_df <- train_df %>%
  select(-any_of("topic")) %>%
  left_join(topic_lookup, by = "post_id")

# Aggiungi topic a test_df
test_df <- test_df %>%
  select(-any_of("topic")) %>%
  left_join(topic_lookup, by = "post_id")

# Allinea i livelli del factor tra train e test
topic_levels <- levels(topic_lookup$topic)

train_df <- train_df %>%
  mutate(topic = factor(topic, levels = topic_levels))

test_df <- test_df %>%
  mutate(topic = factor(topic, levels = topic_levels))

mod_random_lme <- lme (
  share_C1_pre~
    location+
    year+
    topic_YouTube +
    adv_density_last30d+
    season +
    question_marks +
    topic_Private_Life +
    mentions_creators +
    same_topic_last_10posts +
    mentions_singer,
  weights = varIdent(form = ~1|year),
  random = ~1|topic,
  data = train_df
)

summary(mod_random_lme)

vc <- VarCorr(mod_random_lme)

# Variance of the residuals (epsilon)
var_eps = as.numeric(vc[2,1])
var_eps
# Variance of the random effect
var_b = as.numeric(vc[1,1])
var_b

PVRE <- var_b/(var_b+var_eps)
PVRE 

# Random effects: b_0i for i=1,...,n
re = ranef(mod_random_lme)
dat = data.frame(x= row.names(re),y=re[,attr(re,'effectName')])
# The dotplot shows the point and interval estimates for the random effects
# ordered
dotplot(reorder(x,y)~y,data=dat)

#### Comparison between models ####

test_df <- test_df %>%
  mutate(
    n_total = n_C1_pre + n_non_C1_pre,
    share_obs = n_C1_pre / n_total
  ) %>%
  filter(n_total > 0)

fix_to_train_levels <- function(x, train_x) {
  
  train_levels <- if (is.factor(train_x)) {
    levels(train_x)
  } else {
    sort(unique(as.character(train_x)))
  }
  
  x_chr <- trimws(as.character(x))
  x_low <- tolower(x_chr)
  
  # Caso variabili booleane TRUE/FALSE
  if (all(train_levels %in% c("FALSE", "TRUE"))) {
    
    x_rec <- case_when(
      x_low %in% c("true", "t", "1", "yes", "y", "si", "sì") ~ "TRUE",
      x_low %in% c("false", "f", "0", "no", "n") ~ "FALSE",
      TRUE ~ NA_character_
    )
    
    return(factor(x_rec, levels = train_levels))
  }
  
  # Caso factor normali
  factor(x_chr, levels = train_levels)
}

factor_vars <- c(
  "year",
  "season",
  "location",
  "mentions_other_people",
  "has_love_emoji",
  "section"
)

for (v in factor_vars) {
  test_df[[v]] <- fix_to_train_levels(test_df[[v]], train_df[[v]])
}

# Controllo NA generati da livelli non riconosciuti
na_factor_check <- sapply(test_df[factor_vars], function(x) sum(is.na(x)))
na_factor_check

if (any(na_factor_check > 0)) {
  stop("Ci sono ancora NA nei factor del test set. Controlla i livelli con table(test_df$variabile, useNA = 'ifany').")
}


test_df$pred_final <- predict(
  mod_final,
  newdata = test_df,
  type = "response"
)

test_df$pred_gls <- predict(
  mod_gls2,
  newdata = test_df
)

# Clip per evitare valori fuori da [0,1]
test_df$pred_final_clip <- pmin(pmax(test_df$pred_final, 1e-6), 1 - 1e-6)
test_df$pred_gls_clip <- pmin(pmax(test_df$pred_gls, 1e-6), 1 - 1e-6)


# 5. Baseline


baseline_pred <- sum(train_df$n_C1_pre) /
  sum(train_df$n_C1_pre + train_df$n_non_C1_pre)

test_df$pred_baseline <- baseline_pred


# 6. Metriche pesate


metriche <- function(obs, pred, w) {
  data.frame(
    RMSE = sqrt(weighted.mean((obs - pred)^2, w = w)),
    MAE = weighted.mean(abs(obs - pred), w = w),
    Bias = weighted.mean(obs - pred, w = w)
  )
}

perf_final <- metriche(
  obs = test_df$share_obs,
  pred = test_df$pred_final_clip,
  w = test_df$n_total
)

perf_gls <- metriche(
  obs = test_df$share_obs,
  pred = test_df$pred_gls_clip,
  w = test_df$n_total
)

perf_baseline <- metriche(
  obs = test_df$share_obs,
  pred = test_df$pred_baseline,
  w = test_df$n_total
)

perf_table <- bind_rows(
  cbind(model = "mod_final", perf_final),
  cbind(model = "mod_gls_homo", perf_gls),
  cbind(model = "baseline", perf_baseline)
)

perf_table


# 7. Log-loss sui conteggi


logloss_counts <- function(k, n, p) {
  p <- pmin(pmax(p, 1e-6), 1 - 1e-6)
  
  -sum(
    k * log(p) + (n - k) * log(1 - p)
  ) / sum(n)
}

logloss_final <- logloss_counts(
  k = test_df$n_C1_pre,
  n = test_df$n_total,
  p = test_df$pred_final_clip
)

logloss_gls <- logloss_counts(
  k = test_df$n_C1_pre,
  n = test_df$n_total,
  p = test_df$pred_gls_clip
)

logloss_baseline <- logloss_counts(
  k = test_df$n_C1_pre,
  n = test_df$n_total,
  p = test_df$pred_baseline
)

logloss_table <- data.frame(
  model = c("mod_final", "mod_gls_homo", "baseline"),
  logloss = c(logloss_final, logloss_gls, logloss_baseline)
)

logloss_table



# 9. Grafico observed vs predicted


test_long <- test_df %>%
  select(
    post_id,
    share_obs,
    pred_final_clip,
    pred_gls_clip,
    pred_baseline,
    n_total
  ) %>%
  pivot_longer(
    cols = c(pred_final_clip, pred_gls_clip, pred_baseline),
    names_to = "model",
    values_to = "prediction"
  )

ggplot(test_long, aes(x = prediction, y = share_obs)) +
  geom_point(aes(size = n_total), alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  facet_wrap(~ model) +
  theme_minimal() +
  labs(
    title = "Test set: observed vs predicted C1 share",
    x = "Predicted C1 share",
    y = "Observed C1 share",
    size = "N commenters"
  )
