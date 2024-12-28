# Question Answering via intrinsic contextual follow-up questioning
This repo proposes a model architecture which prompts the user with a follow up question for better context on the query by deciding internally wheather to ask a question or not. This is done by combining multiple moodules with different tasks in a single pipeline to ensure better answering in closed generative question answering task.

## Model Architecture
For this study, we have employed FLAN-T5 model as the baseline to generate better answers. More details about the model and ablation studies is briefed in `Milestones data/CS505_Closed_Generative_QA.pdf`

## Data Extraction BigQuery
The dataset has been extracted from the Stack Exchange Data Explorer. The queries utilized in the extraction are given below. All the extracted datasets are listed under the `Datasets` folder
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

## Future of this Study
 - One of the key issues observed is the lack of quality data for the model to train on. We have already accquired some policy data on Visa and Immigration policies of 45 different countries and it is mentioned as `DEMIG` dataset under the `Datasets` folder. We plan to utilize this to pretrain the data for better performance.
 - The question generation model in the current setup is been fulfiled by ChatGPT due to time constraints on the project. We wish to train a different model for this purpose instead of current system.
## Credits and Acknowledgements
This project has been developed as part of CS 505 Introduction to Natural Langugage processing under the guidance of [Professor Andrew Wood](https://www.bu.edu/cs/profiles/andrew-wood-2/) at Boston University.

For any suggestions and ideas on the project, feel free to reach out to us at chvskch@bu.edu or mujindal@bu.edu.
