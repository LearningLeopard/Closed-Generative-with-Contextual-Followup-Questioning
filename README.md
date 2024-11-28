# CS505-NLP-Research-Q-and-A
This repo consists of all the resources utilized for CS 505 NLP course research paper

## Data Extraction BigQuery
### [Version - 1](https://data.stackexchange.com/travel/query/1875546/visa-questions)
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
### [Version - 2](https://data.stackexchange.com/travel/query/1875545/visa-question-and-answers)
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