# Question Answering via intrinsic contextual follow-up questioning
This repo proposes a model architecture which prompts the user with a follow up question for better context on the query by deciding internally wheather to ask a question or not. This is done by combining multiple moodules with different tasks in a single pipeline to ensure better answering in closed generative question answering task.

## Data Extraction BigQuery
The dataset has been extracted from the Stack Exchange Data Explorer. The queries utilized in the extraction are given below. All the extracted datasets are listed under the `Datasets` folder
## [Version - 1](https://data.stackexchange.com/travel/query/1875546/visa-questions)
```
SELECT TOP 10
  q.Score AS [QScore],
  q.Id AS [Post Link],
  q.Title,
  q.Tags,
  q.Body AS [Question Body],
  q.OwnerUserId AS [Questioner],
  a.Score AS [AScore],
  a.Body AS [Answer Body],
  a.CreationDate AS [AnswerDate],
  CASE 
    WHEN q.AcceptedAnswerId = a.Id THEN 1
    ELSE 0
  END AS [IsAcceptedAnswer]
FROM
  Posts AS q
    JOIN Posts AS a
      ON a.ParentId = q.Id
WHERE
  q.Body LIKE '%visa%' 
  OR q.Body LIKE '%immigration%';

```
## [Version - 2](https://data.stackexchange.com/travel/query/1875545/visa-question-and-answers)
```
SELECT
  q.Score AS [QScore],
  q.Id AS [Post Link],
  q.Title,
  q.Tags,
  q.Body AS [Question Body],
  q.OwnerUserId AS [Questioner],
  a.Score AS [AScore],
  a.Body AS [Answer Body],
  a.CreationDate AS [AnswerDate],
  CASE 
    WHEN q.AcceptedAnswerId = a.Id THEN 1
    ELSE 0
  END AS [IsAcceptedAnswer]
FROM
  Posts AS q
    JOIN Posts AS a
      ON a.ParentId = q.Id
WHERE
  q.Body LIKE '%visa%' 
  OR q.Body LIKE '%immigration%'
  OR q.Body LIKE '%citizenship%'
  OR q.Body LIKE '%green card%'
  OR q.Body LIKE '%passport%';

```
# Credits and Acknowledgements
This project has been developed as part of CS 505 Introduction to Natural Langugage processing under the guidance of Professor Andrew Wood at Boston University.
