library(tidyverse)
library(highcharter)

df = read_csv("analysis/backtest-feb.csv", col_types = cols(market_id = col_character())) %>% select(-X1)




# Laying Favourite -------------------------------------------------------

df %>%
  group_by(market_id) %>%
  mutate(
    favPL = case_when(
      bsp == min(bsp) ~ ifelse(win == 1, -(bsp-1), 1),
      TRUE ~ 0
    )
  ) %>%
  ungroup() %>%
  summarise(
    pl = sum(favPL),
    stake = sum(favPL != 0),
    pot = pl / stake
  )

# Flat Staking Hub Ratings -------------------------------------------------------

df %>%
  rename(market_odds = bsp) %>%
  mutate(
    bl = ifelse(model_odds < market_odds, "B", "L"),
    stake = 1,
    pl = case_when(
      bl == "B" ~ ifelse(win == 1, (market_odds-1) * stake, -stake),
      bl == "L" ~ ifelse(win == 0, stake, -(market_odds-1))
    )
  ) %>%
  group_by(bl) %>%
  summarise(
    pl = sum(pl),
    stake = sum(stake),
    pot = pl / stake
  )


# Kelly Staking Hub Ratings -------------------------------------------------------

df %>%
  rename(market_odds = bsp) %>%
  mutate(
    model_prob = 1 / model_odds,
    market_prob = 1 / market_odds
  ) %>%
  mutate(
    bl = ifelse(model_odds < market_odds, "B", "L"),
    stake = case_when(
      bl == "B" ~ (model_prob - market_prob) / (1 - market_prob),
      bl == "L" ~ (market_prob - model_prob) / (1 - market_prob),
    ),
    pl = case_when(
      bl == "B" ~ ifelse(win == 1, (market_odds-1) * stake, -stake),
      bl == "L" ~ ifelse(win == 0, stake, -(market_odds-1))
    )
  ) %>%
  group_by(bl) %>%
  summarise(
    pl = sum(pl),
    stake = sum(stake),
    pot = pl / stake
  )