#!/bin/bash

configurations=(
  "DTNN 2020 0"
  "DCPH 2020 0"
  "BC 2020 0"
  "BC 2018 0"
  "BC 2020 5"
)

for configuration in "${configurations[@]}"; do
  model=$(echo "$configuration" | awk '{print $1}')
  yob_cutoff=$(echo "$configuration" | awk '{print $2}')
  followup_cutoff=$(echo "$configuration" | awk '{print $3}')

  echo "===$configuration ==="
  echo "yob"
  python3 subgroup_yob.py --model "$model" --yob_cutoff "$yob_cutoff" --followup_cutoff "$followup_cutoff"

  echo "follow up"
  python3 subgroup_followup_len.py --model "$model" --yob_cutoff "$yob_cutoff" --followup_cutoff "$followup_cutoff"

  echo "event count"
  python3 subgroup_event_count.py --model "$model" --yob_cutoff "$yob_cutoff" --followup_cutoff "$followup_cutoff"

  echo "sex"
  python3 subgroup_sex.py --model "$model" --yob_cutoff "$yob_cutoff" --followup_cutoff "$followup_cutoff"

  echo "race"
  python3 subgroup_race.py --model "$model" --yob_cutoff "$yob_cutoff" --followup_cutoff "$followup_cutoff"

  echo "insurance"
  python3 subgroup_insurance.py --model "$model" --yob_cutoff "$yob_cutoff" --followup_cutoff "$followup_cutoff"

done
