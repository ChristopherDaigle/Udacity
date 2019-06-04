/* SQL Aggregations */
SELECT
  COUNT(*)
FROM
  accounts;

SELECT
  COUNT(id)
FROM
  accounts;

/* Dealing with NULL s */
SELECT
  COUNT(primary_poc)
FROM
  accounts
WHERE
  primary_poc IS NULL;

/* Quiz */
SELECT
  SUM(poster_qty)
FROM
  orders;

SELECT
  SUM(standard_qty)
FROM
  orders;

SELECT
  SUM(total_amt_usd)
FROM
  orders;

SELECT
  SUM(standard_amt_usd) + SUM(gloss_amt_usd)
FROM
  orders;

-- Can also be solved as (no aggregator):
SELECT
  standard_amt_usd + gloss_amt_usd
FROM
  orders;

SELECT
  ROUND(SUM(standard_amt_usd) / SUM(standard_qty),2)
FROM
  orders;

/* MIN and MAX */
--Quiz
SELECT
  MIN(occurred_at)
FROM
  orders;

SELECT
  occurred_at
FROM
  orders
ORDER BY
  occurred_at
LIMIT
  1;

SELECT
  MAX(occurred_at)
FROM
  web_events;

SELECT
  occurred_at
FROM
  web_events
ORDER BY
  occurred_at DESC
LIMIT
  1;

SELECT
  ROUND(AVG(standard_amt_usd),2) AS AVGstdAMT,
  ROUND(AVG(standard_qty),2) AS AVGstdQTY,
  ROUND(AVG(gloss_amt_usd),2) AS AVGglossAMT,
  ROUND(AVG(gloss_qty),2) AS AVGglossQTY,
  ROUND(AVG(poster_amt_usd),2) AS AVGposterAMT,
  ROUND(AVG(poster_qty),2) AS AVGposterQTY
FROM
  orders;

-- SELECT
--   SUM(*) / 2
-- FROM
--   (SELECT
--   	*
--   FROM(
--     SELECT
--        total_amt_usd
--     FROM
--        orders
--     ORDER BY
--        total_amt_usd
--     LIMIT (SELECT COUNT(*)/2 + 1  FROM orders)) AS T1
--   ORDER BY
--   	total_amt_usd DESC
--   LIMIT
--   	2) AS T2;
SELECT *
FROM (SELECT total_amt_usd
      FROM orders
      ORDER BY total_amt_usd
      LIMIT 3457) AS Table1
ORDER BY total_amt_usd DESC
LIMIT 2;
