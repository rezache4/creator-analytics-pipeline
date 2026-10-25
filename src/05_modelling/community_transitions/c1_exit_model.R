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
#read the csv file at the same directory folder and set the appropriate data types for each column
df <- read.csv(
  "C1_retention_model.csv"
)

df$post_id <- as.factor(df$post_id)
df$is_weekend <- as.logical(df$is_weekend)
df$hour_band <- as.factor(df$hour_band)
df$media_type <- as.factor(df$media_type)

df$mentions_alice <- as.logical(df$mentions_alice)
df$mentions_guglielmo <- as.logical(df$mentions_guglielmo)
df$mentions_creators <- as.logical(df$mentions_creators)
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

library(stringr)

# colonne dummy generate da fastDummies per topic
topic_cols <- grep("^topic_", names(df_rf), value = TRUE)

df_rf <- df_rf %>%
  rowwise() %>%
  mutate(
    topic = {
      hits <- topic_cols[c_across(all_of(topic_cols)) == 1]
      if (length(hits) == 0) {
        NA_character_
      } else {
        str_remove(hits[1], "^topic_")
      }
    }
  ) %>%
  ungroup()

df_rf <- df_rf %>%
  mutate(
    n_C1_total = n_C1_stay + n_C1_exit,
    share_C1_stay = n_C1_stay / n_C1_total
  ) %>%
  filter(n_C1_total >= 1)
##### Train-test split ####
set.seed(123)
train_idx <- createDataPartition(df_rf$share_C1_stay, p = 0.7, list = FALSE)

train_df <- df_rf[train_idx, ]
test_df  <- df_rf[-train_idx, ]


# Define formula (target ~ features)
rf_formula <- share_C1_exit ~ year + month + is_weekend + hour_band + hour + media_type+
  mentions_alice+ mentions_guglielmo+ mentions_creators+ mentions_claudio +
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
  topic_Travel + topic_YouTube + topic

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
  n.var = 35,            # show top 
  type = 1               # type=1 → mean decrease in accuracy
)


#### Correlation ####

num_vars <- c(
  "share_C1_stay", "year", "month", "hour",
  "caption_length", "avg_words_per_sentence",
  "exclamation_marks", "question_marks",
  "posting_frequency_last7d", "days_since_last_post",
  "days_since_last_adv", "adv_density_last30d",
  "same_topic_last_10posts", "n_hashtags",
  "video_duration_filled"
)

df_num <- df_rf[, num_vars]


GGally::ggcorr(df_num, label = TRUE, hjust = 0.9, size = 4)

ggpairs(df_num)


# Histogram of c1 staying percentage
ggplot(df_rf, aes(x = share_C1_stay)) +
  geom_histogram(bins = 50, alpha = 0.6) +
  theme_minimal()

# Histogram of c1 percentage by media type
ggplot(df_rf, aes(x = share_C1_stay, fill = section)) +
  geom_histogram(bins = 50, alpha = 0.6) +
  facet_wrap(~section) +
  theme_minimal()

# Boxplot of c1 stay percentage by section
ggplot(df_rf, aes(share_C1_stay, fill = section)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(title = "C1_stay % by section")

#### Boxplots with mentions ####
# Boxplot of c1 percentage by has_close_friend
ggplot(df_rf, aes(
  y = factor(has_close_friend, labels = c("No", "Yes")),
  x = share_C1_stay,
  fill = factor(has_close_friend, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_stay % by has_close_friend",
    y = "Has Close friend",
    x = "C1_stay percentage",
    fill = "Has close friend"
  )

# Boxplot of c1 percentage by mentions_aimone
ggplot(df_rf, aes(
  y = factor(mentions_aimone, labels = c("No", "Yes")),
  x = share_C1_stay,
  fill = factor(mentions_aimone, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_stay % by mentions_aimone",
    y = "Mentions aimone",
    x = "C1_stay percentage",
    fill = "Mentions aimone"
  )

# Boxplot of c1 percentage by mentions_brand
ggplot(df_rf, aes(
  y = factor(mentions_brand, labels = c("No", "Yes")),
  x = share_C1_stay,
  fill = factor(mentions_brand, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_stay % by mentions_brand",
    y = "Mentions brand",
    x = "C1_stay percentage",
    fill = "Mentions brand"
  )

# Boxplot of c1 percentage by mentions_fashion
ggplot(df_rf, aes(
  y = factor(mentions_fashion, labels = c("No", "Yes")),
  x = share_C1_stay,
  fill = factor(mentions_fashion, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_stay % by mentions_fashion",
    y = "Mentions fashion",
    x = "C1_stay percentage",
    fill = "Mentions fashion"
  )

# Boxplot of c1 percentage by mentions_famous
ggplot(df_rf, aes(
  y = factor(mentions_famous, labels = c("No", "Yes")),
  x = share_C1_stay,
  fill = factor(mentions_famous, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_stay % by mentions_famous",
    y = "Mentions famous",
    x = "C1_stay percentage",
    fill = "Mentions famous"
  )

# Boxplot of c1 percentage by mentions_singer
ggplot(df_rf, aes(
  y = factor(mentions_singer, labels = c("No", "Yes")),
  x = share_C1_stay,
  fill = factor(mentions_singer, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_stay % by mentions_singer",
    y = "Mentions singer",
    x = "C1_stay percentage",
    fill = "Mentions singer"
  )

##### Boxplot of posts characteristics #####

# Boxplot of c1 percentage by is_adv
ggplot(df_rf, aes(
  y = factor(is_adv, labels = c("No", "Yes")),
  x = share_C1_stay,
  fill = factor(is_adv, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_stay % by is_adv",
    y = "Is_adv",
    x = "C1_stay percentage",
    fill = "Is_adv"
  )

# Boxplot of c1 percentage by is_sanremo
ggplot(df_rf, aes(
  y = factor(is_sanremo, labels = c("No", "Yes")),
  x = share_C1_stay,
  fill = factor(is_sanremo, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_stay % by is_sanremo",
    y = "Is_sanremo",
    x = "C1_stay percentage",
    fill = "Is_sanremo"
  )

# Boxplot of c1 percentage by topic (excluding Uncertain, Dance and Other)
ggplot(
  subset(df_rf, !topic %in% c("Uncertain", "Dance", "Other")),
  aes(share_C1_stay, fill = topic)
) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(title = "C1_stay % by topic")

# Boxplot of c1 percentage by location
ggplot(df_rf, aes(
  y = factor(location, labels = c("fake", "no", "real")),
  x = share_C1_stay,
  fill = factor(location, labels = c("fake", "no", "real"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_stay % by location",
    y = "Location",
    x = "C1_stay percentage",
    fill = "Location"
  )

# Boxplot of c1 percentage by audio_type
ggplot(df_rf, aes(
  y = factor(audio_type, labels = c("licensed", "original", "no")),
  x = share_C1_stay,
  fill = factor(audio_type, labels = c("licensed", "original", "no"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_stay % by audio_type",
    y = "Audio type",
    x = "C1_stay percentage",
    fill = "audio type"
  )

# Boxplot of c1 percentage by has_call_to_action
ggplot(df_rf, aes(
  y = factor(has_call_to_action),
  x = share_C1_stay,
  fill = factor(has_call_to_action)
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_stay % by has_call_to_action",
    y = "has_call_to_action",
    x = "C1_stay percentage",
    fill = "has_call_to_action"
  )

# Boxplot of c1 percentage by has_lol_emoji
ggplot(df_rf, aes(
  y = factor(has_lol_emoji),
  x = share_C1_stay,
  fill = factor(has_lol_emoji)
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_stay % by has_lol_emoji",
    y = "has_lol_emoji",
    x = "C1_stay percentage",
    fill = "has_lol_emoji"
  )

# Boxplot of c1 percentage by has_shine_emoji
ggplot(df_rf, aes(
  y = factor(has_shine_emoji),
  x = share_C1_stay,
  fill = factor(has_shine_emoji)
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_stay % by has_shine_emoji",
    y = "has_shine_emoji",
    x = "C1_stay percentage",
    fill = "has_shine_emoji"
  )

###### Boxplot of posts time settings ######

# Boxplot of c1 percentage by hour_band
ggplot(df_rf, aes(
  y = factor(hour_band, labels = c("afternoon", "evening", "morning")),
  x = share_C1_stay,
  fill = factor(hour_band, labels = c("afternoon", "evening", "morning"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1_stay % by hour_band",
    y = "Hour_band",
    x = "C1_stay percentage",
    fill = "hour band"
  )



# Boxplot C1_stay % by season and year
ggplot(df_rf, aes(
  x = season,
  y = share_C1_stay,
  fill = season
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_wrap(~ season_year, nrow = 1) +
  theme_minimal() +
  labs(
    title = "C1_stay % by season and year",
    x = "Season",
    y = "C1_stay percentage",
    fill = "Season"
  )

ggplot(df_rf, aes(
  x = season,
  y = share_C1_stay,
  fill = season
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_grid(section ~ season_year) +
  theme_minimal() +
  labs(
    title = "C1_stay % by season, year and section",
    x = "Season",
    y = "C1_stay percentage",
    fill = "Season"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom"
  )

# Boxplot of c1 percentage by year and section
bw1 <- bwplot(share_C1_stay~ year_f | section, data = df_rf)
xlims <- c("2023","2024","2025", "2026")
update(bw1, xlim = xlims, pch = "|")


# Boxplot of c1 percentage by year and section
bw2 <- bwplot(share_C1_stay~ as.factor(month) | as.factor(year), data = df_rf)
xlims <- c("gen","feb","mar", "apr", "may","jun","jul", "aug", "sep","oct","nov", "dec")
update(bw2, xlim = xlims, pch = "|")


# Boxplot of c1 percentage by year and section
bw3 <- bwplot(share_C1_stay~ season | as.factor(year), data = df_rf)
xlims <- c("spring", "summer", "autumn", "winter")
update(bw3, xlim = xlims, pch = "|")

#### Beta-Binomial model ####

library(glmmTMB)
train_df <- train_df %>%
  mutate(
    n_total = n_C1_stay + n_C1_exit,
    year = factor(year),
    month = factor(month),
    season = factor(season),
    hour_band = factor(hour_band),
    hour = factor(hour), 
    location = factor(location),
    section = factor(section),
    audio_type = factor(audio_type),
    topic = factor(topic)
  ) %>%
  filter(n_total > 0)

test_df <- test_df %>%
  mutate(
    n_total = n_C1_stay + n_C1_exit,
    year = factor(year),
    month = factor(month),
    season = factor(season),
    hour_band = factor(hour_band),
    hour = factor(hour),
    location = factor(location),
    section = factor(section),
    audio_type = factor(audio_type), 
    topic = factor(topic)
  ) %>%
  filter(n_total > 0)

# Modeling exit risk for C1
mod_bb_exit <- glmmTMB(
  cbind(n_C1_exit, n_C1_stay) ~
    mentions_alice +
    caption_length +
    season+
    mentions_aimone +
    topic_Theatre, 
  family = betabinomial(),
  data = train_df
)

summary(mod_bb_exit)

mod_no_theatre <- update(
  mod_bb_exit,
  . ~ . - topic_Theatre
)

anova(mod_no_theatre, mod_bb_exit)
# Keep topic_Theatre

set.seed(123)

res_final <- simulateResiduals(
  fittedModel = mod_bb_exit,
  n = 1000
)

plot(res_final)

testUniformity(res_final)
testDispersion(res_final)
testZeroInflation(res_final)
testOutliers(res_final)

# Find the outlier
# DHARMa scaled residuals
sr <- res_final$scaledResiduals

# Exact boundary residuals: potential DHARMa outliers
outlier_idx <- which(sr == 0 | sr == 1)

# DHARMa scaled residuals
sr <- res_final$scaledResiduals

# Exact boundary residuals: potential DHARMa outliers
outlier_idx <- which(sr == 0 | sr == 1)

train_df[outlier_idx, ]
# Post-id = 18074117857510337 

library(performance)
check_collinearity(mod_bb_exit)

plotResiduals(res_final, form = train_df$caption_length)


train_df$share_obs <- train_df$n_C1_exit / 
  (train_df$n_C1_stay + train_df$n_C1_exit)

train_df$share_pred <- predict(mod_bb_exit, type = "response")

ggplot(train_df, aes(x = share_pred, y = share_obs)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  theme_minimal() +
  labs(
    title = "Observed vs predicted C1 share",
    x = "Predicted share C1",
    y = "Observed share C1"
  )

# Because C1 exit is a rare event, post-level observed exit shares are highly 
# discrete and noisy. Model calibration was therefore assessed by grouping posts 
# into bins of predicted exit probability and comparing the weighted observed 
# exit rate with the mean predicted probability

test_calib <- test_df %>%
  mutate(
    n_total_exit = n_C1_exit + n_C1_stay,
    obs_exit_share = n_C1_exit / n_total_exit,
    pred_exit_share = predict(mod_bb_exit, newdata = test_df, type = "response"),
    pred_bin = ntile(pred_exit_share, 5)
  ) %>%
  group_by(pred_bin) %>%
  summarise(
    mean_pred = weighted.mean(pred_exit_share, n_total_exit, na.rm = TRUE),
    obs_rate = sum(n_C1_exit, na.rm = TRUE) / sum(n_total_exit, na.rm = TRUE),
    n_posts = n(),
    n_users = sum(n_total_exit, na.rm = TRUE),
    .groups = "drop"
  )

ggplot(test_calib, aes(x = mean_pred, y = obs_rate, size = n_users)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(
    x = "Mean predicted C1 exit probability",
    y = "Observed C1 exit rate",
    size = "C1 users",
    title = "Calibration plot for C1 exit model"
  ) +
  theme_minimal()
# The calibration plot shows that predicted and observed exit risks are broadly 
# aligned, although some deviations remain due to the low frequency of exit events.

#### ODDS RATIO ####

coefs <- summary(mod_bb_exit)$coefficients$cond

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
# Be cautious with mentions_aimone because it is true 3 times only

#### Binomial check ####

mod_bin_exit <- glmmTMB(
  cbind(n_C1_exit, n_C1_stay) ~
    mentions_alice +
    caption_length +
    season +
    mentions_aimone,
  family = binomial(),
  data = train_df
)

AIC(mod_bb_exit, mod_bin_exit)
BIC(mod_bb_exit, mod_bin_exit)
