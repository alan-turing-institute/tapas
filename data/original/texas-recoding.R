## Recoding of the "Texas" dataset

require(tidyverse)

texas <- read_csv("texas-orig.csv",
                  col_names = TRUE,
                  col_types = "_cccccccccccccccccc")

## DISCHARGE
## ---------
## Original: quarters from "2013Q1" to "2013Q4"
## New: Integers

quarters_to_integers <- function(quarters) {
    yr <- as.numeric(str_sub(quarters, 1L, 4L))
    qr <- as.numeric(str_sub(quarters, -1L, -1L))
    (yr - 2013) * 4 + (qr - 1)
    }

texas$DISCHARGE <- quarters_to_integers(texas$DISCHARGE)

## ADMIT_WEEKDAY
## -------------
## Original: 1, 2, ..., 7 (NB: "INVALID" is not a possible value)
## New: 0, 1, ..., 6

texas$ADMIT_WEEKDAY = as.numeric(texas$ADMIT_WEEKDAY) - 1

## PAT_AGE
## -------
## Original: Naturals, with the possibility of "INVALID", which, however, does
## not occur in the data.
## New: Naturals

## No actual recoding required.

## LENGTH_OF_STAY
## --------------
## Original: 1, 2, ..., 986
## New: Naturals (ie, allowing zero as well)

## No actual recoding required.

## Output csv file
## ---------------

## Omit header row
## Omit row numbers 

write_csv(texas, "texas-new.csv",
          col_names = FALSE)
