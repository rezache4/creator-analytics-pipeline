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
  "C1_C3_transition_model.csv",
  colClasses = c(post_id = "character")
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
    n_C1_C2_exit_total = n_C1_C2_to_C3 + n_C1_C2_exit_not_C3,
    share_C1_C2_to_C3 = n_C1_C2_to_C3 / n_C1_C2_exit_total
  ) %>%
  filter(n_C1_C2_exit_total >= 2)

set.seed(123)
train_idx <- createDataPartition(df_rf$share_C1_C2_to_C3, p = 0.7, list = FALSE)

train_df <- df_rf[train_idx, ]
test_df  <- df_rf[-train_idx, ]


# Define formula (target ~ features)
rf_formula <- share_C1_C2_to_C3 ~ year + month + is_weekend + hour_band + hour + media_type+
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

#### Plots ####

# Histogram of moves to c3 staying percentage
ggplot(df_rf, aes(x = share_C1_C2_to_C3)) +
  geom_histogram(bins = 50, alpha = 0.6) +
  theme_minimal()

# Histogram of moves to c3 by media type
ggplot(df_rf, aes(x = share_C1_C2_to_C3, fill = section)) +
  geom_histogram(bins = 50, alpha = 0.6) +
  facet_wrap(~section) +
  theme_minimal()

# Boxplot of moves to c3 percentage by section
ggplot(df_rf, aes(share_C1_C2_to_C3, fill = section)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(title = "C1+ C2 moves to c3  by section")

#### Boxplots with mentions ####
# Boxplot of moves to c3 by has_close_friend
ggplot(df_rf, aes(
  y = factor(has_close_friend, labels = c("No", "Yes")),
  x = share_C1_C2_to_C3,
  fill = factor(has_close_friend, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1+ C2 moves to c3  by has_close_friend",
    y = "Has Close friend",
    x = "C1+ C2 moves to c3 percentage",
    fill = "Has close friend"
  )

# Boxplot of moves to c3 by mentions_aimone
ggplot(df_rf, aes(
  y = factor(mentions_aimone, labels = c("No", "Yes")),
  x = share_C1_C2_to_C3,
  fill = factor(mentions_aimone, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1+ C2 moves to c3  by mentions_aimone",
    y = "Mentions aimone",
    x = "C1+ C2 moves to c3 percentage",
    fill = "Mentions aimone"
  )

# Boxplot of moves to c3 by mentions_brand
ggplot(df_rf, aes(
  y = factor(mentions_brand, labels = c("No", "Yes")),
  x = share_C1_C2_to_C3,
  fill = factor(mentions_brand, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1+ C2 moves to c3  by mentions_brand",
    y = "Mentions brand",
    x = "C1+ C2 moves to c3 percentage",
    fill = "Mentions brand"
  )

# Boxplot of moves to c3 by mentions_fashion
ggplot(df_rf, aes(
  y = factor(mentions_fashion, labels = c("No", "Yes")),
  x = share_C1_C2_to_C3,
  fill = factor(mentions_fashion, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1+ C2 moves to c3  by mentions_fashion",
    y = "Mentions fashion",
    x = "C1+ C2 moves to c3 percentage",
    fill = "Mentions fashion"
  )

# Boxplot of moves to c3 by mentions_famous
ggplot(df_rf, aes(
  y = factor(mentions_famous, labels = c("No", "Yes")),
  x = share_C1_C2_to_C3,
  fill = factor(mentions_famous, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1+ C2 moves to c3  by mentions_famous",
    y = "Mentions famous",
    x = "C1+ C2 moves to c3 percentage",
    fill = "Mentions famous"
  )

# Boxplot of moves to c3 by mentions_singer
ggplot(df_rf, aes(
  y = factor(mentions_singer, labels = c("No", "Yes")),
  x = share_C1_C2_to_C3,
  fill = factor(mentions_singer, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1+ C2 moves to c3  by mentions_singer",
    y = "Mentions singer",
    x = "C1+ C2 moves to c3 percentage",
    fill = "Mentions singer"
  )

##### Boxplot of posts characteristics #####

# Boxplot of moves to c3 by is_adv
ggplot(df_rf, aes(
  y = factor(is_adv, labels = c("No", "Yes")),
  x = share_C1_C2_to_C3,
  fill = factor(is_adv, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1+ C2 moves to c3  by is_adv",
    y = "Is_adv",
    x = "C1+ C2 moves to c3 percentage",
    fill = "Is_adv"
  )

# Boxplot of moves to c3 by is_sanremo
ggplot(df_rf, aes(
  y = factor(is_sanremo, labels = c("No", "Yes")),
  x = share_C1_C2_to_C3,
  fill = factor(is_sanremo, labels = c("No", "Yes"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1+ C2 moves to c3  by is_sanremo",
    y = "Is_sanremo",
    x = "C1+ C2 moves to c3 percentage",
    fill = "Is_sanremo"
  )

# Boxplot of moves to c3 by topic (excluding Uncertain, Dance and Other)
ggplot(
  subset(df_rf, !topic %in% c("Uncertain", "Dance", "Other")),
  aes(share_C1_C2_to_C3, fill = topic)
) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(title = "C1+ C2 moves to c3  by topic")

# Boxplot of moves to c3 by location
ggplot(df_rf, aes(
  y = factor(location, labels = c("fake", "no", "real")),
  x = share_C1_C2_to_C3,
  fill = factor(location, labels = c("fake", "no", "real"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1+ C2 moves to c3  by location",
    y = "Location",
    x = "C1+ C2 moves to c3 percentage",
    fill = "Location"
  )

# Boxplot of moves to c3 by audio_type
ggplot(df_rf, aes(
  y = factor(audio_type, labels = c("licensed", "original", "no")),
  x = share_C1_C2_to_C3,
  fill = factor(audio_type, labels = c("licensed", "original", "no"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1+ C2 moves to c3  by audio_type",
    y = "Audio type",
    x = "C1+ C2 moves to c3 percentage",
    fill = "audio type"
  )

# Boxplot of moves to c3 by has_call_to_action
ggplot(df_rf, aes(
  y = factor(has_call_to_action),
  x = share_C1_C2_to_C3,
  fill = factor(has_call_to_action)
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1+ C2 moves to c3  by has_call_to_action",
    y = "has_call_to_action",
    x = "C1+ C2 moves to c3 percentage",
    fill = "has_call_to_action"
  )

# Boxplot of moves to c3 by has_lol_emoji
ggplot(df_rf, aes(
  y = factor(has_lol_emoji),
  x = share_C1_C2_to_C3,
  fill = factor(has_lol_emoji)
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1+ C2 moves to c3  by has_lol_emoji",
    y = "has_lol_emoji",
    x = "C1+ C2 moves to c3 percentage",
    fill = "has_lol_emoji"
  )

# Boxplot of moves to c3 by has_shine_emoji
ggplot(df_rf, aes(
  y = factor(has_shine_emoji),
  x = share_C1_C2_to_C3,
  fill = factor(has_shine_emoji)
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1+ C2 moves to c3  by has_shine_emoji",
    y = "has_shine_emoji",
    x = "C1+ C2 moves to c3 percentage",
    fill = "has_shine_emoji"
  )

###### Boxplot of posts time settings ######

# Boxplot of moves to c3 by hour_band
ggplot(df_rf, aes(
  y = factor(hour_band, labels = c("afternoon", "evening", "morning")),
  x = share_C1_C2_to_C3,
  fill = factor(hour_band, labels = c("afternoon", "evening", "morning"))
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "C1+ C2 moves to c3  by hour_band",
    y = "Hour_band",
    x = "C1+ C2 moves to c3 percentage",
    fill = "hour band"
  )



# Boxplot C1+ C2 moves to c3 by season and year
ggplot(df_rf, aes(
  x = season,
  y = share_C1_C2_to_C3,
  fill = season
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_wrap(~ season_year, nrow = 1) +
  theme_minimal() +
  labs(
    title = "C1+ C2 moves to c3 by season and year",
    x = "Season",
    y = "C1+ C2 moves to c3percentage",
    fill = "Season"
  )

ggplot(df_rf, aes(
  x = season,
  y = share_C1_C2_to_C3,
  fill = season
)) +
  geom_boxplot(outlier.alpha = 0.3) +
  facet_grid(section ~ season_year) +
  theme_minimal() +
  labs(
    title = "C1+ C2 moves to c3 by season, year and section",
    x = "Season",
    y = "C1+ C2 moves to c3 percentage",
    fill = "Season"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom"
  )

# Boxplot of moves to c3 by year and section
bw1 <- bwplot(share_C1_C2_to_C3~ year_f | section, data = df_rf)
xlims <- c("2023","2024","2025", "2026")
update(bw1, xlim = xlims, pch = "|")


# Boxplot of moves to c3 by year and section
bw2 <- bwplot(share_C1_C2_to_C3~ as.factor(month) | as.factor(year), data = df_rf)
xlims <- c("gen","feb","mar", "apr", "may","jun","jul", "aug", "sep","oct","nov", "dec")
update(bw2, xlim = xlims, pch = "|")


# Boxplot of moves to c3 by year and section
bw3 <- bwplot(share_C1_C2_to_C3~ season | as.factor(year), data = df_rf)
xlims <- c("spring", "summer", "autumn", "winter")
update(bw3, xlim = xlims, pch = "|")

#### Beta-Binomial model ####

library(glmmTMB)
prep_c3_data <- function(df) {
  df %>%
    mutate(
      n_C1_C2_exit_total = n_C1_C2_to_C3 + n_C1_C2_exit_not_C3,
      
      year = factor(year),
      month = factor(month),
      
      season = factor(
        as.character(season),
        levels = c("winter", "spring", "summer", "autumn")
      ),
      
      hour_band = factor(
        as.character(hour_band),
        levels = c("morning", "afternoon", "evening")
      ),
      
      audio_type = factor(
        as.character(audio_type),
        levels = c("", "licensed_music", "original_sounds")
      ),
      
      topic_Theatre = factor(
        as.character(topic_Theatre),
        levels = c("FALSE", "TRUE")
      ),
      
      mentions_alice = factor(
        as.character(mentions_alice),
        levels = c("FALSE", "TRUE")
      ),
      
      mentions_guglielmo = factor(
        as.character(mentions_guglielmo),
        levels = c("FALSE", "TRUE")
      ),
      
      location = factor(location),
      section = factor(section),
      hour = factor(hour)
    ) %>%
    filter(n_C1_C2_exit_total > 0)
}

train_df <- prep_c3_data(train_df)
test_df  <- prep_c3_data(test_df)

mod_bb_complex <- glmmTMB(
  cbind(n_C1_C2_to_C3, n_C1_C2_exit_not_C3) ~
    season +
    avg_words_per_sentence +
    topic_Theatre +
    mentions_alice +
    mentions_creators +
    audio_type +
    is_supplied +
    mentions_guglielmo, 
  family = betabinomial(),
  data = train_df
)

summary(mod_bb_complex)

# Removed is_gifted, has_shine_emoji, topic_Other, topic_Travel " dropping columns from rank-deficient conditional model"
# Removed has_lol_emoji, topic_YouTube, mentions_famous, adv_density_last30d, 
# mentions_singer, question_marks, topic_Cinema_and_TV, topic_Podcast, mentions_fashion
# video_duration_filled, has_love_emoji, posting_frequency_last7d, mentions_other_people
# topic_Food,is_weekend, 


mod_no_supplied <- update(
  mod_bb_complex,
  . ~ . - is_supplied
)

anova(mod_no_supplied, mod_bb_complex)
# Remove is_supplied
summary(mod_no_supplied)
# Remove mentions_creators

mod_bb_simple <- glmmTMB(
  cbind(n_C1_C2_to_C3, n_C1_C2_exit_not_C3) ~
    season +
    avg_words_per_sentence +
    topic_Theatre +
    mentions_alice +
    audio_type +
    mentions_guglielmo, 
  family = betabinomial(),
  data = train_df
)

summary(mod_bb_simple)

set.seed(123)

res_final <- simulateResiduals(
  fittedModel = mod_bb_simple,
  n = 1000
)

plot(res_final)
# Check for model convergence
mod_bb_simple$sdr$pdHess
# True -> Positive definite
mod_bb_simple$fit$convergence
# 0 -> convergence

testUniformity(res_final)
testDispersion(res_final)
testZeroInflation(res_final)
testOutliers(res_final)


library(performance)
check_collinearity(mod_bb_simple)

train_df$share_obs <- train_df$n_C1_C2_to_C3 / 
  (train_df$n_C1_C2_to_C3 + train_df$n_C1_C2_exit_not_C3)

train_df$share_pred <- predict(mod_bb_simple, type = "response")

ggplot(train_df, aes(x = share_pred, y = share_obs)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  theme_minimal() +
  labs(
    title = "Observed vs predicted C1 share",
    x = "Predicted share C1",
    y = "Observed share C1"
  )

rhs_form <- ~ season +
  avg_words_per_sentence +
  topic_Theatre +
  mentions_alice +
  audio_type +
  mentions_guglielmo

factor_vars <- c(
  "season",
  "topic_Theatre",
  "mentions_alice",
  "audio_type",
  "mentions_guglielmo"
)

contr_train <- lapply(train_df[factor_vars], contrasts)

X_test <- model.matrix(
  rhs_form,
  data = test_df,
  contrasts.arg = contr_train
)

beta <- fixef(mod_bb_simple)$cond

# Controllo colonne
setdiff(names(beta), colnames(X_test))
setdiff(colnames(X_test), names(beta))

# Allinea le colonne nell'ordine dei coefficienti
X_test <- X_test[, names(beta), drop = FALSE]

pred_test <- plogis(drop(X_test %*% beta))
test_calib_c3 <- test_df %>%
  mutate(
    n_total_c3 = n_C1_C2_to_C3 + n_C1_C2_exit_not_C3,
    pred_c3_share = pred_test
  ) %>%
  filter(n_total_c3 > 0) %>%
  mutate(
    obs_c3_share = n_C1_C2_to_C3 / n_total_c3,
    pred_bin = ntile(pred_c3_share, 5)
  ) %>%
  group_by(pred_bin) %>%
  summarise(
    mean_pred = weighted.mean(pred_c3_share, n_total_c3, na.rm = TRUE),
    obs_rate = sum(n_C1_C2_to_C3, na.rm = TRUE) / sum(n_total_c3, na.rm = TRUE),
    n_posts = n(),
    n_users = sum(n_total_c3, na.rm = TRUE),
    .groups = "drop"
  )

ggplot(test_calib_c3, aes(x = mean_pred, y = obs_rate, size = n_users)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(
    x = "Mean predicted probability of transition to C3",
    y = "Observed C1/C2-to-C3 transition rate",
    size = "C1/C2 exit users",
    title = "Calibration plot for C1/C2 → C3 model"
  ) +
  theme_minimal()
