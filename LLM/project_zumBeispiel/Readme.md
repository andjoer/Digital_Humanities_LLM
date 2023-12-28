
# Annotation only synthetic
## Model: gpt-4-1106-preview
Evaluation dataset: full human annotated Dataset

### gpt4_human_annotated_examples_v3_wiedergabe_gedankengang

|                    | erzählstimme_nein | figurenrede_ja | figurenrede_nein | erzählstimme_ja |
|--------------------|-------------------|----------------|------------------|-----------------|
| erzählstimme_nein  |                12 |              2 |                3 |              14 |
| figurenrede_ja     |                 0 |              0 |                2 |               2 |
| figurenrede_nein   |                14 |              0 |               56 |              10 |
| erzählstimme_ja    |                 6 |              1 |                0 |              38 |

#### F1 Scores:
- erzählstimme_nein: 0.38
- figurenrede_ja: 0.00
- figurenrede_nein: 0.79
- erzählstimme_ja: 0.70


### gpt4_human_annotated_examples_v3_wiedergabe_erzählposition


|                          | erzählstimme_auktorial | figurenrede_personal | figurenrede_auktorial | erzählstimme_personal |
|--------------------------|------------------------|----------------------|-----------------------|-----------------------|
| erzählstimme_auktorial   |                     19 |                    2 |                     2 |                     4 |
| figurenrede_personal     |                      8 |                   34 |                    13 |                     9 |
| figurenrede_auktorial    |                      9 |                    0 |                    11 |                     0 |
| erzählstimme_personal    |                      8 |                    1 |                     1 |                    39 |

#### F1 Scores:
- erzählstimme_auktorial: 0.54
- figurenrede_personal: 0.67
- figurenrede_auktorial: 0.47
- erzählstimme_personal: 0.77


### gpt4_human_annotated_examples_v3_erzählposition_gedankengang


|                | auktorial_ja | personal_nein | personal_ja | auktorial_nein |
|----------------|--------------|---------------|-------------|----------------|
| auktorial_ja   |            0 |             0 |           0 |              2 |
| personal_nein  |            2 |            33 |           9 |             22 |
| personal_ja    |            4 |             4 |          37 |              2 |
| auktorial_nein |            9 |             0 |           6 |             30 |

#### F1 Scores:
- auktorial_ja: 0.00
- personal_nein: 0.64
- personal_ja: 0.75
- auktorial_nein: 0.59


### gpt4_human_annotated_examples_v3_erzählposition


|           | auktorial | personal |
|-----------|-----------|----------|
| auktorial |        41 |        6 |
| personal  |        30 |       83 |

#### F1 Scores:
- auktorial: 0.69
- personal: 0.82


### gpt4_human_annotated_examples_v3_wiedergabeform


|              | erzählstimme | figurenrede |
|--------------|--------------|-------------|
| erzählstimme |           70 |           6 |
| figurenrede  |           26 |          58 |

#### F1 Scores:
- erzählstimme: 0.81
- figurenrede: 0.78


### gpt4_human_annotated_examples_v3_gedankengang


|      | ja | nein |
|------|----|------|
| ja   | 41 |    8 |
| nein | 26 |   85 |

#### F1 Scores:
- ja: 0.71
- nein: 0.83

## Model: gpt-3.5-turbo-1106
Dataset: only 1600 synthetic
Evaluation dataset: full human annotated Dataset

### gpt4_human_annotated_examples_v3_wiedergabe_gedankengang

|                    | figurenrede_nein | erzählstimme_ja | figurenrede_ja | erzählstimme_nein |
|--------------------|------------------|-----------------|----------------|-------------------|
| figurenrede_nein   |               50 |              11 |              0 |                20 |
| erzählstimme_ja    |                1 |              39 |              2 |                 3 |
| figurenrede_ja     |                3 |               1 |              0 |                 0 |
| erzählstimme_nein  |                2 |               8 |              2 |                19 |

#### F1 Scores:
- figurenrede_nein: 0.73
- erzählstimme_ja: 0.75
- figurenrede_ja: 0.00
- erzählstimme_nein: 0.52

### gpt4_human_annotated_examples_v3_wiedergabe_erzählposition

|                          | erzählstimme_auktorial | erzählstimme_personal | figurenrede_personal | figurenrede_auktorial |
|--------------------------|------------------------|-----------------------|----------------------|-----------------------|
| erzählstimme_auktorial   |                     19 |                     8 |                    0 |                     0 |
| erzählstimme_personal    |                      3 |                    39 |                    7 |                     0 |
| figurenrede_personal     |                      6 |                    14 |                   41 |                     4 |
| figurenrede_auktorial    |                     11 |                     1 |                    4 |                     4 |

#### F1 Scores:
- erzählstimme_auktorial: 0.58
- erzählstimme_personal: 0.70
- figurenrede_personal: 0.70
- figurenrede_auktorial: 0.29


### gpt4_human_annotated_examples_v3_erzählposition_gedankengang

|                | personal_nein | auktorial_nein | personal_ja | auktorial_ja |
|----------------|---------------|----------------|-------------|--------------|
| personal_nein  |            44 |             10 |          12 |            1 |
| auktorial_nein |             7 |             30 |           6 |            2 |
| personal_ja    |             4 |              1 |          41 |            1 |
| auktorial_ja   |             0 |              2 |           0 |            0 |

#### F1 Scores:
- personal_nein: 0.72
- auktorial_nein: 0.68
- personal_ja: 0.77
- auktorial_ja: 0.00

### gpt4_human_annotated_examples_v3_erzählposition

|           | auktorial | personal |
|-----------|-----------|----------|
| auktorial |        34 |       13 |
| personal  |        13 |      101 |

#### F1 Scores:
- auktorial: 0.72
- personal: 0.89

### gpt4_human_annotated_examples_v3_wiedergabeform

|              | erzählstimme | figurenrede |
|--------------|--------------|-------------|
| erzählstimme |           69 |           7 |
| figurenrede  |           32 |          53 |

#### F1 Scores:
- erzählstimme: 0.78
- figurenrede: 0.73

### gpt4_human_annotated_examples_v3_gedankengang


|      | nein | ja |
|------|------|----|
| nein |   91 | 21 |
| ja   |    7 | 42 |

#### F1 Scores:
- nein: 0.87
- ja: 0.75

# Annotation synthetic plus 57 human annotated
## Model: gpt-4-1106-preview
Evaluationdataset: Human annotated minus the training Data for GPT3

### human_annotated_eval_gpt3_wiedergabe_gedankengang

|                    | erzählstimme_ja | figurenrede_nein | figurenrede_ja | erzählstimme_nein |
|--------------------|-----------------|------------------|----------------|-------------------|
| erzählstimme_ja    |              24 |                0 |              2 |                 3 |
| figurenrede_nein   |               4 |               37 |              0 |                13 |
| figurenrede_ja     |               1 |                1 |              0 |                 0 |
| erzählstimme_nein  |               8 |                3 |              2 |                12 |

#### F1 Scores:
- erzählstimme_ja: 0.73
- figurenrede_nein: 0.78
- figurenrede_ja: 0.00
- erzählstimme_nein: 0.45
- _nein: 1.00

### human_annotated_eval_gpt3_wiedergabe_erzählposition


|                          | erzählstimme_auktorial | figurenrede_personal | figurenrede_auktorial | erzählstimme_personal |
|--------------------------|------------------------|----------------------|-----------------------|-----------------------|
| erzählstimme_auktorial   |                     14 |                    1 |                     2 |                     6 |
| figurenrede_personal     |                      6 |                   22 |                    10 |                     5 |
| figurenrede_auktorial    |                      7 |                    3 |                     3 |                     0 |
| erzählstimme_personal    |                      3 |                    4 |                     0 |                    24 |

#### F1 Scores:
- erzählstimme_auktorial: 0.53
- figurenrede_personal: 0.60
- figurenrede_auktorial: 0.21
- erzählstimme_personal: 0.73


### human_annotated_eval_gpt3_erzählposition_gedankengang

|                | auktorial_ja | personal_ja | personal_nein | auktorial_nein |
|----------------|--------------|-------------|---------------|----------------|
| auktorial_ja   |            0 |           1 |             0 |              1 |
| personal_ja    |            1 |          25 |             2 |              1 |
| personal_nein  |            1 |           5 |            24 |             16 |
| auktorial_nein |            4 |           4 |             5 |             21 |

#### F1 Scores:
- auktorial_ja: 0.00
- personal_ja: 0.78
- personal_nein: 0.62
- auktorial_nein: 0.58


### human_annotated_eval_gpt3_erzählposition


|           | auktorial | personal |
|-----------|-----------|----------|
| auktorial |        26 |       10 |
| personal  |        19 |       56 |

#### F1 Scores:
- auktorial: 0.64
- personal: 0.79

### human_annotated_eval_gpt3_wiedergabeform


|              | figurenrede | erzählstimme |
|--------------|-------------|--------------|
| figurenrede  |          38 |           18 |
| erzählstimme |           7 |           47 |

#### F1 Scores:
- figurenrede: 0.75
- erzählstimme: 0.79

### human_annotated_eval_gpt3_gedankengang


|      | ja | nein |
|------|----|------

|
| ja   | 27 |    4 |
| nein | 14 |   66 |

#### F1 Scores:
- ja: 0.75
- nein: 0.88



## Model: gpt-3.5-turbo-1106
Dataset: 450 synthetic plus 57 human annotated

### human_annotated_eval_gpt3_wiedergabe_gedankengang


|                    | figurenrede_nein | erzählstimme_ja | erzählstimme_nein | figurenrede_ja |
|--------------------|------------------|-----------------|-------------------|----------------|
| figurenrede_nein   |               33 |               5 |                17 |              0 |
| erzählstimme_ja    |                0 |              27 |                 1 |              1 |
| erzählstimme_nein  |                1 |               7 |                16 |              0 |
| figurenrede_ja     |                2 |               0 |                 0 |              0 |

#### F1 Scores:
- figurenrede_nein: 0.73
- erzählstimme_ja: 0.79
- erzählstimme_nein: 0.55
- figurenrede_ja: 0.00


### human_annotated_eval_gpt3_wiedergabe_erzählposition


|                          | erzählstimme_personal | figurenrede_personal | erzählstimme_auktorial | figurenrede_auktorial |
|--------------------------|-----------------------|----------------------|------------------------|-----------------------|
| erzählstimme_personal    |                    28 |                    2 |                      1 |                     0 |
| figurenrede_personal     |                     9 |                   31 |                      3 |                     1 |
| erzählstimme_auktorial   |                     6 |                    0 |                     16 |                     0 |
| figurenrede_auktorial    |                     0 |                    3 |                     10 |                     0 |

#### F1 Scores:
- erzählstimme_personal: 0.76
- figurenrede_personal: 0.78
- erzählstimme_auktorial: 0.62
- figurenrede_auktorial: 0.00


### human_annotated_eval_gpt3_erzählposition_gedankengang


|                | auktorial_ja | personal_ja | auktorial_nein | personal_nein |
|----------------|--------------|-------------|----------------|---------------|
| auktorial_ja   |            0 |           1 |              1 |             0 |
| personal_ja    |            0 |          27 |              0 |             2 |
| auktorial_nein |            1 |           4 |             24 |             4 |
| personal_nein  |            0 |           7 |              5 |            34 |

#### F1 Scores:
- auktorial_ja: 0.00
- personal_ja: 0.79
- auktorial_nein: 0.76
- personal_nein: 0.79


### human_annotated_eval_gpt3_erzählposition


|           | personal | auktorial |
|-----------|----------|-----------|
| personal  |       70 |         5 |
| auktorial |        9 |        26 |

#### F1 Scores:
- personal: 0.91
- auktorial: 0.79


### human_annotated_eval_gpt3_wiedergabeform


|              | erzählstimme | figurenrede |
|--------------|--------------|-------------|
| erzählstimme |           51 |           2 |
| figurenrede  |           22 |          35 |

#### F1 Scores:
- erzählstimme: 0.81
- figurenrede: 0.74


### human_annotated_eval_gpt3_gedankengang


|      | ja | nein |
|------|----|------|
| ja   | 28 |    3 |
| nein | 12 |   67 |

#### F1 Scores:
- ja: 0.79
- nein: 0.90

## Model: gpt-3.5-turbo-1106
Dataset: 1600 synthetic plus 57 human annotated

### human_annotated_eval_gpt3_wiedergabe_gedankengang

|                    | erzählstimme_ja | erzählstimme_nein | figurenrede_nein | figurenrede_ja |
|--------------------|-----------------|-------------------|------------------|----------------|
| erzählstimme_ja    |              26 |                 2 |                0 |              1 |
| erzählstimme_nein  |               6 |                17 |                1 |              1 |
| figurenrede_nein   |               7 |                13 |               35 |              0 |
| figurenrede_ja     |               0 |                 0 |                2 |              0 |

#### F1 Scores:
- erzählstimme_ja: 0.76
- erzählstimme_nein: 0.60
- figurenrede_nein: 0.75
- figurenrede_ja: 0.00


### human_annotated_eval_gpt3_wiedergabe_erzählposition


|                          | figurenrede_auktorial | erzählstimme_auktorial | figurenrede_personal | erzählstimme_personal |
|--------------------------|-----------------------|------------------------|----------------------|-----------------------|
| figurenrede_auktorial    |                      2 |                      8 |                    3 |                     0 |
| erzählstimme_auktorial   |                      0 |                     16 |                    0 |                     7 |
| figurenrede_personal     |                      4 |                      3 |                   28 |                     9 |
| erzählstimme_personal    |                      0 |                      2 |                    3 |                    26 |

#### F1 Scores:
- figurenrede_auktorial: 0.21
- erzählstimme_auktorial: 0.62
- figurenrede_personal: 0.72
- erzählstimme_personal: 0.71


### human_annotated_eval_gpt3_erzählposition_gedankengang


|                | auktorial_nein | personal_nein | auktorial_ja | personal_ja |
|----------------|----------------|---------------|--------------|-------------|
| auktorial_nein |             24 |             5 |            1 |           4 |
| personal_nein  |              8 |            29 |            0 |           9 |
| auktorial_ja   |              1 |             0 |            0 |           1 |
| personal_ja    |              1 |             2 |            0 |          26 |

#### F1 Scores:
- auktorial_nein: 0.71
- personal_nein: 0.71
- auktorial_ja: 0.00
- personal_ja: 0.75


### human_annotated_eval_gpt3_erzählposition


|           | personal | auktorial |
|-----------|----------|-----------|
| personal  |       66 |         9 |
| auktorial |       10 |        26 |

#### F1 Scores:
- personal: 0.87
- auktorial: 0.73


### human_annotated_eval_gpt3_wiedergabeform


|              | erzählstimme | figurenrede |
|--------------|--------------|-------------|
| erzählstimme |           51 |           3 |
| figurenrede  |           20 |          37 |

#### F1 Scores:
- erzählstimme: 0.82
- figurenrede: 0.76


### human_annotated_eval_gpt3_gedankengang

|      | nein | ja |
|------|------|----|
| nein |   66 | 14 |
| ja   |    4 | 27 |

#### F1 Scores:
- nein: 0.88
- ja: 0.75

## Model: gpt-3.5-turbo-1106
Dataset: only 1600 synthetic

### finetune_gpt3_clean_eval_wiedergabe_gedankengang


|                    | figurenrede_ja | figurenrede_nein | erzählstimme_nein | erzählstimme_ja |
|--------------------|----------------|------------------|-------------------|-----------------|
| figurenrede_ja     |              0 |                0 |                 0 |               0 |
| figurenrede_nein   |              0 |                9 |                 1 |               4 |
| erzählstimme_nein  |              3 |                2 |                62 |               4 |
| erzählstimme_ja    |              4 |                1 |                11 |              33 |

#### F1 Scores:
- figurenrede_ja: 0.00
- figurenrede_nein: 0.69
- erzählstimme_nein: 0.86
- erzählstimme_ja: 0.73


### finetune_gpt3_clean_eval_wiedergabe_erzählposition


|                          | erzählstimme_personal | figurenrede_personal | figurenrede_auktorial | erzählstimme_auktorial |
|--------------------------|-----------------------|----------------------|-----------------------|------------------------|
| erzählstimme_personal    |                     38 |                    5 |                     0 |                      5 |
| figurenrede_personal     |                      4 |                    7 |                     1 |                      0 |
| figurenrede_auktorial    |                      0 |                    1 |                     0 |                      1 |
| erzählstimme_auktorial   |                      0 |                    3 |                     2 |                     67 |

#### F1 Scores:
- erzählstimme_personal: 0.84
- figurenrede_personal: 0.50
- figurenrede_auktorial: 0.00
- erzählstimme_auktorial: 0.92


### finetune_gpt3_clean_eval_erzählposition_gedankengang

|                | auktorial_nein | auktorial_ja | personal_nein | personal_ja |
|----------------|----------------|--------------|---------------|-------------|
| auktorial_nein |             60 |            1 |             1 |           2 |
| auktorial_ja   |              7 |            2 |             0 |           1 |
| personal_nein  |              3 |            0 |            10 |           8 |
| personal_ja    |              3 |            0 |             2 |          34 |

#### F1 Scores:
- auktorial_nein: 0.88
- auktorial_ja: 0.31
- personal_nein: 0.59
- personal_ja: 0.81

### finetune_gpt3_clean_eval_erzählposition

|           | personal | auktorial |
|-----------|----------|-----------|
| personal  |       54 |         6 |
| auktorial |        4 |        70 |

#### F1 Scores:
- personal: 0.92
- auktorial: 0.93


### finetune_gpt3_clean_eval_wiedergabeform


|              | figurenrede | erzählstimme |
|--------------|-------------|--------------|
| figurenrede  |           9 |            5 |
| erzählstimme |          10 |          110 |

#### F1 Scores:
- figurenrede: 0.55
- erzählstimme: 0.94


### finetune_gpt3_clean_eval_gedankengang


|      | ja | nein |
|------|----|------|
| ja   | 37 |   12 |
| nein | 11 |   74 |

#### F1 Scores:
- ja: 0.76
- nein: 0.87


# Classification of examples into categories

|    | 4c | 1a | 4b | 2a | 4a | 3b | 2b | 4d | 4e | 3a |
|----|----|----|----|----|----|----|----|----|----|----|
| 4c |  1 |  0 |  0 |  1 |  0 |  0 |  0 |  0 |  0 |  0 |
| 1a |  2 | 10 |  0 |  9 |  0 |  5 |  1 |  0 |  0 |  1 |
| 4b |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
| 2a |  3 |  1 |  4 | 64 |  3 | 11 |  2 |  0 |  0 |  4 |
| 4a |  0 |  0 |  1 |  5 |  1 |  0 |  0 |  0 |  0 |  0 |
| 3b |  0 |  2 |  1 |  4 |  0 |  3 |  1 |  0 |  0 |  1 |
| 2b |  1 |  1 |  1 |  2 |  0 |  2 |  5 |  0 |  0 |  0 |
| 4d |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  2 |  0 |  0 |
| 4e |  2 |  1 |  0 |  1 |  0 |  0 |  0 |  0 |  1 |  0 |
| 3a |  0 |  0 |  2 | 16 |  0 |  4 |  2 |  0 |  0 |  7 |

#### F1 Scores:
- 4c: 0.18
- 1a: 0.47
- 4b: 0.00
- 2a: 0.66
- 4a: 0.18
- 3b: 0.16
- 2b: 0.43
- 4d: 1.00
- 4e: 0.33
- 3a: 0.32


