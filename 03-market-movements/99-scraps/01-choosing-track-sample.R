library(tidyverse)

odds = read_csv("/media/hdd/tmp/thoroughbred-parsed/thoroughbred-odds-2021.csv", col_types = cols(market_id = "c", selection_id = "c"))
races = read_csv("/media/hdd/tmp/thoroughbred-parsed/thoroughbred-race-data.csv", col_types = cols(market_id = "c", selection_id = "c"))


odds %>%
  select(market_id, selection_id, traded_vol) %>%
  inner_join(
    races %>% select(market_id, selection_id, track)
  ) %>%
  group_by(track) %>%
  summarise(
    runners = n(),
    markets = n_distinct(market_id),
    volume = sum(traded_vol)
  ) %>%
  mutate(
    avg_volume = volume / markets
  ) %>%
  arrange(desc(avg_volume)) %>%
  print(n = 20)

# big vic ones I can see: Flemington, Caulfield, Moonee Valley, Sandown, Bendigo 