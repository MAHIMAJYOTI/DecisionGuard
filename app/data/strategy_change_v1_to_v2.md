# Strategy change: v1 to v2

## Summary

Credit v2 loosens the bureau cutoff from 600 to 580 and the DTI cutoff from 50% to 55%, and tightens the thin-file score bands. Fraud v2 turns a 1-hour velocity burst into a challenge instead of a block, and raises the model challenge cutoff from 0.35 to 0.40. Both challengers use the same saved models as the champions.

Recommendation: **NO-GO**.

- Credit v2 is NO-GO: observed swap-in bad rate 0.0824 is more than 1.25 times the champion approval observed bad rate 0.0580.
- Expected loss moves -0.9% (52,191,011 to 51,733,947 rupees). LGD is the 0.45 planning assumption, not an observed loss.
- Fraud v2 is CONDITIONAL GO: fraud caught changes by -0.0169, inside 2 percentage points, and genuine blocks fall from 2,410 to 2,311. The condition is that a CHALLENGE on a velocity burst is an acceptable customer step.

## Credit model, old split versus matured split

Maturity cutoff: application date on or before 2025-10-01 (as-of 2026-10-01 minus 12 months). Known-outcome rows 17,273. Matured rows 15,649.

| | old split | matured split (saved model) |
| --- | ---: | ---: |
| train bad rate | 0.0759 | 0.0761 |
| test bad rate | 0.2000 | 0.0716 |
| AUC | 0.7059 | 0.7064 |
| Gini | 0.4119 | 0.4128 |
| KS | 0.3169 | 0.3077 |
| train n | 13,818 | 12,519 |
| test n | 3,455 | 3,130 |

The old test bad rate is higher because a loan can be labelled bad from month 3 and good only after month 12. The newest known rows are mostly early defaults. The saved model drops every application whose 12-month window is still open, then splits out-of-time inside what remains.

### Calibration (predicted PD vs actual bad rate)

| decile | old predicted | old actual | matured predicted | matured actual |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0194 | 0.0751 | 0.0190 | 0.0064 |
| 2 | 0.0299 | 0.0754 | 0.0292 | 0.0224 |
| 3 | 0.0386 | 0.1040 | 0.0372 | 0.0575 |
| 4 | 0.0472 | 0.1333 | 0.0451 | 0.0288 |
| 5 | 0.0576 | 0.1156 | 0.0538 | 0.0479 |
| 6 | 0.0697 | 0.2174 | 0.0652 | 0.0703 |
| 7 | 0.0850 | 0.2087 | 0.0790 | 0.0863 |
| 8 | 0.1045 | 0.2601 | 0.0960 | 0.0799 |
| 9 | 0.1363 | 0.3536 | 0.1252 | 0.1182 |
| 10 | 0.2281 | 0.4566 | 0.2014 | 0.1981 |

## Credit swap-set matrix

Rows are credit_v1. Columns are credit_v2.

| v1 \ v2 | APPROVE | REFER | DECLINE |
| --- | ---: | ---: | ---: |
| APPROVE | 19,398 | 747 | 0 |
| REFER | 0 | 9,273 | 220 |
| DECLINE | 321 | 350 | 19,057 |

## Credit impact

Expected loss uses model PD x LGD 0.45 x approved limit. assumed share of the approved limit lost if the loan reaches 90+ DPD.

| version | approval rate | refer rate | decline rate | expected bad rate of approvals | observed bad rate of approvals | labelled approvals | expected loss (INR) | refers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| credit_v1 | 40.81% | 19.23% | 39.96% | 0.0442 | 0.0580 | 10,727 | 52,191,011 | 9,493 |
| credit_v2 | 39.94% | 21.01% | 39.05% | 0.0446 | 0.0587 | 10,485 | 51,733,947 | 10,370 |

Referral volume change (v2 minus v1): +877.

## Swap-ins and swap-outs

Expected bad rate is the average model PD. Observed bad rate uses only rows the historical book labelled good or bad.

| slice | count | average model PD (expected) | expected bad rate on labelled rows | expected bad count (sum of PD) | observed bad rate | labelled rows | observed bads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| swap-ins: v1 DECLINE or REFER, v2 APPROVE | 321 | 0.0547 | 0.0550 | 9.3542 | 0.0824 | 170 | 14 |
| swap-outs: v1 APPROVE, v2 not APPROVE | 747 | 0.0394 | 0.0392 | 16.1456 | 0.0510 | 412 | 21 |

## Why each swap happened

Swap-ins are DECLINE or REFER to APPROVE. Swap-outs are APPROVE to anything else. The other columns are decision changes that do not cross the approve line.

| cause | swap-ins | swap-outs | DECLINE to APPROVE | DECLINE to REFER | REFER to APPROVE | REFER to DECLINE | APPROVE to REFER | APPROVE to DECLINE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bureau_cutoff_600_to_580 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dti_cutoff_50_to_55 | 321 | 0 | 321 | 350 | 0 | 0 | 0 | 0 |
| bureau_cutoff_and_dti_cutoff | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| thin_file_score_band | 0 | 747 | 0 | 0 | 0 | 220 | 747 | 0 |
| other_reason_change | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Bureau scores 580-599: 732 applications, of which 716 are declined by v1 for the bureau rule. Decision changes in that group: 0. Their mean model PD is 0.7242 and the lowest is 0.1775, both above the 0.15 decline cutoff. 59 of them have no remaining hard-rule decline in v2 and are declined by the model cutoff alone. The bureau loosening therefore changes the reason on some declines and does not create approvals.

## Fraud comparison

| metric | fraud_v1 | fraud_v2 |
| --- | ---: | ---: |
| alert rate | 0.0755 | 0.0575 |
| fraud caught | 0.8736 | 0.8567 |
| fraud value caught (INR) | 27,272,770 | 27,213,247 |
| fraud value blocked (INR) | 26,920,770 | 26,920,770 |
| fraud value challenged (INR) | 352,000 | 292,477 |
| false positives | 35,438 | 26,542 |
| genuine transactions blocked | 2,410 | 2,311 |
| genuine transactions challenged | 33,028 | 24,231 |
| genuine customers blocked | 2,147 | 2,107 |
| genuine customers challenged | 16,500 | 14,482 |
| legitimate grocery-burst transactions | 200 | 200 |
| grocery bursts blocked | 82 | 3 |
| grocery bursts challenged | 1 | 80 |
| grocery bursts allowed | 117 | 117 |
| genuine grocery with 3+ txns in 1 hour | 89 | 89 |
| those velocity rows blocked | 89 | 2 |
| those velocity rows challenged | 0 | 87 |

### Why the fraud rate is higher in the model test window

Oldest 80% of transactions: fraud rate 0.0040 on 398,768 rows, ending 2026-09-13 06:13:07.
Newest 20%: fraud rate 0.0089 on 99,693 rows, starting 2026-09-13 06:13:18.
Before 2026-08-12: 0 frauds in 220,626 rows.
From 2026-08-12 onward: 2,492 frauds in 277,835 rows (rate 0.0090).

Legitimate tickets are spread across all 90 days. Fraud tickets are planted only from day 40, so a 30-day customer history can exist before the fraud. The oldest 80% of rows includes that quiet stretch, so its fraud rate is lower. The newest 20% sits inside the window where the fraud was planted.

## Risks

- Reject inference: the credit model only saw booked loans. A swap-in that the old book never funded has a model PD and no observed outcome.
- Outcome maturity: goods need 12 months. The saved model trains only on applications dated on or before the as-of date minus 12 months. Applications after that date are scored, not used to fit.
- Loss given default is assumed at 0.45. It is not estimated from these files.
- The labelled swap-ins are DTI moves. The old booking policy declined DTI above 60%, so a DTI between 50% and 55% could already have a repayment outcome. Bureau scores 580-599 do not become approvals.
- A grocery burst is identified as five genuine grocery tickets for one customer inside 20 minutes. That is the planted pattern. It is not a column on the mart.
- Fraud value blocked does not change. The drop in fraud value caught is the same size as the drop in fraud value challenged, which is the model challenge cutoff moving from 0.35 to 0.40. A velocity burst whose model score is already at or above 0.80 stays blocked.

## Golden tests

Result: **PASS**. Exit code 0.

```
4 passed in 5.93s
```

## Sign-off

| role | status |
| --- | --- |
| Business owner | pending |
| Risk | pending |
| IT | pending |

Go / no-go recommendation: **NO-GO**.

- Credit challenger: NO-GO
- Fraud challenger: CONDITIONAL GO

Reasons:

- Credit v2 is NO-GO: observed swap-in bad rate 0.0824 is more than 1.25 times the champion approval observed bad rate 0.0580.
- Expected loss moves -0.9% (52,191,011 to 51,733,947 rupees). LGD is the 0.45 planning assumption, not an observed loss.
- Fraud v2 is CONDITIONAL GO: fraud caught changes by -0.0169, inside 2 percentage points, and genuine blocks fall from 2,410 to 2,311. The condition is that a CHALLENGE on a velocity burst is an acceptable customer step.
