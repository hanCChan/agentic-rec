# Phase 1.18h Strategy Prompt V2 Report

- mode: `strategy_prompt_v2`
- target_collapse_only: **False**
- num_groups: **20**
- v1_strategy_collapse_count: **0**
- v2_strategy_collapse_count: **0**
- collapse_fixed_count: **0**
- collapse_fix_rate: **0.00**
- v2_gate_passed: **True**
- phase2_candidate_ready: **True**

## Per-Group Comparison

- `esci_val_0`: v1_unique=4 v2_unique=4 v1_jaccard=0.640 v2_jaccard=0.514 collapse_fixed=False
  - v2 queries: ['bathroom fan without light', 'bathroom fan without light business security', 'bathroom exhaust fan no light', 'bathroom fan no light no illumination']
- `esci_val_1`: v1_unique=2 v2_unique=4 v1_jaccard=0.750 v2_jaccard=0.259 collapse_fixed=False
  - v2 queries: ['!awnmower tires without rims', '!awnmower tires without rims business security adhesive peel seal gummed pack white office mailing', 'tires without rims lawn mower', '!awnmower tires no rim']
- `esci_val_10`: v1_unique=3 v2_unique=4 v1_jaccard=0.811 v2_jaccard=0.594 collapse_fixed=False
  - v2 queries: ['resveratrol complex 60 caps', 'resveratrol complex 60 caps pack', 'resveratrol formula 60 caps', 'resveratrol complex 60 caps no window']
- `esci_val_11`: v1_unique=4 v2_unique=3 v1_jaccard=0.568 v2_jaccard=0.795 collapse_fixed=False
  - v2 queries: ['#10 window envelopes not self seal', '#10 window envelopes not self seal security', '10 window envelopes not self seal', '#10 window envelopes not self seal']
- `esci_val_12`: v1_unique=4 v2_unique=4 v1_jaccard=0.473 v2_jaccard=0.698 collapse_fixed=False
  - v2 queries: ['#4 braiding hair not stretched', '#4 braiding hair not stretched business', '#4 braiding hair not stretched not loose', '#4 braiding hair not pre-stretched']
- `esci_val_13`: v1_unique=4 v2_unique=3 v1_jaccard=0.644 v2_jaccard=0.766 collapse_fixed=False
  - v2 queries: ['overnight pads for women extra heavy without wings', 'overnight pads for women extra heavy without wings office security adhesive', 'overnight pads for women extra heavy without wings', 'overnight pads for women extra heavy no wings']
- `esci_val_14`: v1_unique=3 v2_unique=4 v1_jaccard=0.787 v2_jaccard=0.652 collapse_fixed=False
  - v2 queries: ['10 hour pads without wings', '10 hour pads without wings security', '10 hour back support pads without wings', '10 hour pads no wings']
- `esci_val_15`: v1_unique=3 v2_unique=3 v1_jaccard=0.867 v2_jaccard=0.667 collapse_fixed=False
  - v2 queries: ['maxi pads without wings', 'maxi pads without wings pack', 'maxi pads no wings', 'maxi pads no wings']
- `esci_val_17`: v1_unique=3 v2_unique=4 v1_jaccard=0.583 v2_jaccard=0.614 collapse_fixed=False
  - v2 queries: ['always maxi overnight pads without wings', 'always maxi overnight pads without wings office mailing security', 'maxi overnight pads no wings', 'always maxi overnight pads no wings']
- `esci_val_18`: v1_unique=3 v2_unique=3 v1_jaccard=0.780 v2_jaccard=0.819 collapse_fixed=False
  - v2 queries: ['always maxi pads long super without wings', 'always maxi pads long super without wings pack', 'always maxi pads long super no wings', 'always maxi pads long super without wings']
- `esci_val_19`: v1_unique=3 v2_unique=2 v1_jaccard=0.622 v2_jaccard=0.900 collapse_fixed=False
  - v2 queries: ['#8 tags without string', '#8 tags without string electronics', '#8 tags without string', '#8 tags without string']
- `esci_val_4`: v1_unique=2 v2_unique=2 v1_jaccard=0.857 v2_jaccard=0.714 collapse_fixed=False
  - v2 queries: ['#10 window envelopes without plastic', '#10 window envelopes without plastic', '10 window envelopes no plastic window', '#10 window envelopes without plastic']
- `esci_val_5`: v1_unique=3 v2_unique=4 v1_jaccard=0.795 v2_jaccard=0.706 collapse_fixed=False
  - v2 queries: ['10 open window envelopes without plastic window', '10 open window envelopes without plastic window security', '10 open window envelopes no plastic window', '10 open window envelopes no plastic no seal']
- `esci_val_7`: v1_unique=4 v2_unique=4 v1_jaccard=0.412 v2_jaccard=0.640 collapse_fixed=False
  - v2 queries: ['08 do not disturb', '08 do not disturb office', '08 do not disturb novelty', '08 do not disturb without do not disturb security tint']
- `esci_val_9`: v1_unique=4 v2_unique=2 v1_jaccard=0.632 v2_jaccard=0.818 collapse_fixed=False
  - v2 queries: ['#1 best and not expensive bath back brush cream color', '#1 best and not expensive bath back brush cream color', 'bath back brush cream color not expensive good', '#1 best and not expensive bath back brush cream color']

Proceed to Phase 2.1 Tiny GRPO Smoke Training with strict step limits and no checkpoint promotion.

Strategy prompt V2 rerollout for collapse diagnostics only; no GRPO training was performed.
