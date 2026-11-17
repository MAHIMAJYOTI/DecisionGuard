"""Assumptions used only when comparing strategies.

LGD is not in the data. A booked loan that reaches 90+ DPD does not tell us
how much of the limit was eventually lost. 45% is a planning assumption for
an unsecured personal loan or card balance, and the impact table says so.
"""

LGD = 0.45
