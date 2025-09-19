## 專案原則
- 程式碼必須簡潔，只實作我要求的功能，不要額外添加功能。
- 請勿任意安裝套件 or 建置conda環境。(必須經過我同意。)
- 請勿使用 sudo 權限進行操作。
- 請勿使用專案外的資料。只能使用專案內的程式碼以及資料。
- 推理過程和總結的語言，請使用繁體中文，必要時附上英文術語。

## 任務概述
Subtask 3: Dimensional Aspect Sentiment Quad Prediction (DimASQP)
Given a textual instance, extract all (A, C, O, VA) quadruplets, where A denotes an aspect term, C an aspect category, O an opinion term, and VA a valence-arousal score. This task is an extension of Subtask 2 (triplet extraction), with the addition of the aspect category element. The input is in JSON Lines format and includes the following fields:

"ID" – A unique identifier for the instance.
"Text" – A sentence or paragraph expressing subjective opinions.
The output should be in JSON Lines format and include the following fields. All textual outputs are case-sensitive.

"ID" – Should match the input ID.
"Quadruplet" – A list of extracted quadruplets, where each quadruplet includes the following fields.
"Aspect" – The aspect term (string), which should retain the same case as in the input text.
"Category" – The aspect category (string), formatted as ENTITY#ATTRIBUTE and written in UPPERCASE. For all valid combinations, see the full list of aspect categories.
"Opinion" – The opinion term (string), which should retain the same case as in the input text.
"VA" – The valence-arousal score is a string in V#A format, with each value ranging from 1.00 to 9.00 and rounded to two decimal places.

trial data path: "/workplace/dxlin/Homework/NLP/DimABSA2026/trial_data"